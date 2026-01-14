import logging
import time
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=fp8transformer,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    @torch.inference_mode()
    #@profile
    def __call__(
            self,
            prompt: str,
            seed: int,
            height: int,
            width: int,
            num_frames: int,
            frame_rate: float,
            images: list[tuple[str, int, float]],
            tiling_config: TilingConfig | None = None,
            enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        import hashlib
        import os

        print("Start Call")
        startAt = time.time()
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16
        print("starting text encoder", time.time() - startAt)

        # --- DISK CACHE LOGIC START ---
        CACHE_DIR = "./prompt_embeddings_cache"
        os.makedirs(CACHE_DIR, exist_ok=True)

        # 1. Create a unique hash string based on inputs that affect text encoding
        # Distilled pipeline usually doesn't use negative prompts, so we exclude it from the hash
        image_identifier = images[0][0] if (len(images) > 0 and enhance_prompt) else "no_img"

        hash_input_str = (
            f"prompt:{prompt}|"
            f"pipeline:distilled|"  # Differentiates from standard t2v if they share a folder
            f"enhance:{enhance_prompt}|"
            f"seed:{seed if enhance_prompt else 'ignored'}|"
            f"img:{image_identifier}"
        )

        # Create MD5 hash for filename
        cache_filename = hashlib.md5(hash_input_str.encode('utf-8')).hexdigest() + ".pt"
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        context_p = None

        if os.path.exists(cache_path):
            print(f"Disk cache hit! Loading embeddings from {cache_path}")
            try:
                # Load directly to the correct device
                # For distilled, we only saved context_p
                context_p = torch.load(cache_path, map_location=self.device)
            except Exception as e:
                print(f"Failed to load cache (corrupted?): {e}. Regenerating.")

        # If cache miss or load failed
        if context_p is None:
            print("Disk cache miss. Running text encoder.")
            text_encoder = self.model_ledger.text_encoder()

            # Logic to handle prompt enhancement
            current_prompt = prompt
            if enhance_prompt:
                current_prompt = generate_enhanced_prompt(
                    text_encoder, prompt, images[0][0] if len(images) > 0 else None
                )

            # In distilled pipeline, we usually only take the first element (positive)
            # and there is no negative context generated
            context_p = encode_text(text_encoder, prompts=[current_prompt])[0]

            # Save to disk for next time
            print(f"Saving embeddings to {cache_path}")
            torch.save(context_p, cache_path)

            torch.cuda.synchronize()
            del text_encoder
            cleanup_memory()
        # --- DISK CACHE LOGIC END ---

        # Unpack the positive context (Distilled usually splits this into video/audio context)
        video_context, audio_context = context_p

        print("end text encoder", time.time() - startAt)

        print("Stage 1: Initial low resolution video generation.", time.time() - startAt)
        # Stage 1: Initial low resolution video generation.
        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()

        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop(
                sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        print("Stage 1: Starting denoising loop.", time.time() - startAt)
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )
        print("Stage 1: End denoising loop.", time.time() - startAt)

        print("Stage 2: Upsample and refine the video at higher resolution with distilled LORA.", time.time() - startAt)
        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=self.model_ledger.spatial_upsampler()
        )
        print("Stage 2: Upsample and refine the video end.", time.time() - startAt)

        torch.cuda.synchronize()
        cleanup_memory()

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )
        print("Stage 2: Upsample and refine the video end.", time.time() - startAt)
        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        decoded_video = vae_decode_video(video_state.latent, self.model_ledger.video_decoder(), tiling_config)
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder()
        )
        print("Stage 2:vae decode video end.", time.time() - startAt)
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_distilled_arg_parser()
    args = parser.parse_args()
    pipeline = DistilledPipeline(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
