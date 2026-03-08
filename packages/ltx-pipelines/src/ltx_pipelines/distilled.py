import logging
import time
import hashlib
import os

from collections.abc import Iterator
import numpy as np

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger, euler_denoising_loop
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    get_device,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()

logging.basicConfig(level=logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("ltx_core").setLevel(logging.ERROR)


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=distilled_checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def get_interpolated_sigmas(self, num_steps: int, device: torch.device) -> torch.Tensor:
        original_sigmas = DISTILLED_SIGMA_VALUES
        old_x = np.linspace(0, 1, len(original_sigmas))
        new_x = np.linspace(0, 1, num_steps + 1)
        new_sigmas = np.interp(new_x, old_x, original_sigmas)
        new_sigmas[0] = original_sigmas[0]
        new_sigmas[-1] = original_sigmas[-1]
        return torch.tensor(new_sigmas, dtype=torch.float32, device=device)

    def get_interpolated_sigmas2(self, num_steps: int, device: torch.device) -> torch.Tensor:
        original_sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
        old_x = np.linspace(0, 1, len(original_sigmas))
        new_x = np.linspace(0, 1, num_steps + 1)
        new_sigmas = np.interp(new_x, old_x, original_sigmas)
        new_sigmas[0] = original_sigmas[0]
        new_sigmas[-1] = original_sigmas[-1]
        return torch.tensor(new_sigmas, dtype=torch.float32, device=device)

    @torch.inference_mode()
    def __call__(
            self,
            prompt: str,
            seed: int,
            height: int,
            width: int,
            num_frames: int,
            frame_rate: float,
            images: list[ImageConditioningInput],
            tiling_config: TilingConfig | None = None,
            enhance_prompt: bool = False,
            output_path: str = '',
            video_chunks_number: int = 0,
            fps: int = 0,
            disable_audio: bool = True,
            save_step_1_preview: bool = False,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        print("Preparing Inference")
        startAt = time.time()
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        # --- PROMPT CACHE LOGIC START ---
        CACHE_DIR = "./prompt_embeddings_cache"
        os.makedirs(CACHE_DIR, exist_ok=True)

        image_identifier = images[0][0] if (len(images) > 0 and enhance_prompt) else "no_img"

        hash_input_str = (
            f"prompt:{prompt}|"
            f"pipeline:distilled|"
            f"enhance:{enhance_prompt}|"
            f"seed:{seed if enhance_prompt else 'ignored'}|"
            f"img:{image_identifier}"
        )

        cache_filename = hashlib.md5(hash_input_str.encode('utf-8')).hexdigest() + ".pt"
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        context_p = None
        if os.path.exists(cache_path):
            print(f"Prompt cache hit! Loading embeddings from {cache_path}")
            try:
                context_p = torch.load(cache_path, map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"Failed to load cache (corrupted?): {e}. Regenerating.")

        if context_p is None:
            print("Prompt cache miss. Running text encoder.")
            (context_p,) = encode_prompts(
                [prompt],
                self.model_ledger,
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            )

            print(f"Saving embeddings to {cache_path}")
            torch.save(context_p, cache_path)
            print("Prompt encoded.", time.time() - startAt)

        # --- PROMPT CACHE LOGIC END ---

        video_context, audio_context = context_p.video_encoding, context_p.audio_encoding

        print("Stage 1: Initial low resolution video generation.")
        # Stage 1: Initial low resolution video generation.

        transformer = self.model_ledger.transformer()
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        # stage_1_sigmas = self.get_interpolated_sigmas(16, self.device)

        if not disable_audio:
            pass
        else:
            audio_context = None

        def denoising_loop(
                sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol, is_conditioning: bool = True
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
                    is_conditioning=is_conditioning,
                    disable_audio=disable_audio,
                ),
                disable_audio=disable_audio,
            )
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = []
        is_conditioning = False
        if images:
            is_conditioning = True
            video_encoder = self.model_ledger.video_encoder()
            stage_1_conditionings = combined_image_conditionings(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )
            torch.cuda.synchronize()
            del video_encoder
            cleanup_memory()
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
            is_conditioning=is_conditioning
        )
        print("Stage 1: Finish denoising loop.", time.time() - startAt)
        torch.cuda.synchronize()
        del stage_1_sigmas
        del stage_1_output_shape
        del stage_1_conditionings
        cleanup_memory()

        if save_step_1_preview:
            video_decoder = self.model_ledger.video_decoder()
            decoded_video = vae_decode_video(
                video_state.latent, self.model_ledger.video_decoder(), tiling_config, generator
            )
            torch.cuda.synchronize()
            del video_decoder
            cleanup_memory()
            if not disable_audio:
                vocoder = self.model_ledger.vocoder()
                decoded_audio = vae_decode_audio(
                    audio_state.latent, self.model_ledger.audio_decoder(), vocoder
                )
                torch.cuda.synchronize()
                del vocoder
                cleanup_memory()
            else:
                decoded_audio = None
            encode_video(
                video=decoded_video,
                fps=fps,
                audio=decoded_audio,
                output_path=output_path.replace('.mp4', '_.mp4'),
                video_chunks_number=video_chunks_number,
            )

        print("Stage 2: Upsample and refine the video at higher resolution with distilled LORA.", time.time() - startAt)
        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        video_encoder = self.model_ledger.video_encoder()
        upsampler = self.model_ledger.spatial_upsampler()
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=upsampler
        )
        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        # stage_2_sigmas = self.get_interpolated_sigmas2(10, self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = []
        if images:
            stage_2_conditionings = combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )

        torch.cuda.synchronize()
        del video_encoder
        del upsampler
        del video_state
        cleanup_memory()

        audio_latents = None
        if not disable_audio:
            audio_latents = audio_state.latent

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
            initial_audio_latent=audio_latents,
            is_conditioning=is_conditioning
        )
        print("Stage 2: Finish upsample and refine the video.", time.time() - startAt)
        torch.cuda.synchronize()
        del transformer
        del stage_2_output_shape
        del stage_2_conditionings
        del stage_2_sigmas
        cleanup_memory()
        print("Stage 3: Starting vae decode video.", time.time() - startAt)
        video_decoder = self.model_ledger.video_decoder()
        decoded_video = vae_decode_video(
            video_state.latent, self.model_ledger.video_decoder(), tiling_config, generator
        )
        del video_decoder
        cleanup_memory()

        if not disable_audio:
            vocoder = self.model_ledger.vocoder()
            decoded_audio = vae_decode_audio(
                audio_state.latent, self.model_ledger.audio_decoder(), vocoder
            )
            del vocoder
            cleanup_memory()
        else:
            decoded_audio = None
        print("Stage 3: Done.", time.time() - startAt)
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    args = parser.parse_args()
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
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
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
        fps=args.frame_rate,
        disable_audio=args.disable_audio,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
