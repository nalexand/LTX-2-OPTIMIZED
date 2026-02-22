import logging
import time
import os
import hashlib
import einops

from collections.abc import Iterator
from dataclasses import replace

import torch
import torchaudio

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_music_arg_parser
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    multi_modal_guider_denoising_func,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
    noise_audio_state
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()

logging.basicConfig(level=logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("ltx_core").setLevel(logging.ERROR)

AUDIO_SAMPLE_RATE = 16000


def load_audio_input(audio_path: str, target_sample_rate: int, device: torch.device) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    
    return waveform.to(device)


class MusicToVideoTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline with audio conditioning.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled checkpoint for higher
    quality output. Supports optional image and audio conditioning.
    """

    def __init__(
        self,
        checkpoint_path: str,
        stage_2_checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        print("Start Init")
        startAt = time.time()
        self.device = device
        self.dtype = torch.bfloat16
        
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
        )

        self.stage_2_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=stage_2_checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=[],  # Stage 2 distilled checkpoint doesn't need loras
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        print("End Init", time.time() - startAt)

    def encode_audio_latents(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encodes the audio waveform into latents using the VAE encoder.
        """
        from ltx_core.model.audio_vae.ops import AudioProcessor

        n_fft = 1024
        mel_hop_length = 160
        mel_bins = 64
        
        audio_processor = AudioProcessor(
            sample_rate=AUDIO_SAMPLE_RATE,
            mel_bins=mel_bins,
            mel_hop_length=mel_hop_length,
            n_fft=n_fft
        ).to(self.device)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] > 2:
            waveform = waveform[:, :2, :]
            
        spectrogram = audio_processor.waveform_to_mel(waveform.to(self.device).float(), AUDIO_SAMPLE_RATE)
        audio_encoder = self.stage_1_model_ledger.audio_encoder()
        encoded_latents = audio_encoder(spectrogram.to(self.dtype))

        del audio_encoder
        del audio_processor
        cleanup_memory()
        
        return encoded_latents

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
            self,
            prompt: str,
            negative_prompt: str,
            seed: int,
            height: int,
            width: int,
            num_frames: int,
            frame_rate: float,
            num_inference_steps: int,
            video_guider_params: MultiModalGuiderParams,
            audio_guider_params: MultiModalGuiderParams,
            images: list[tuple[str, int, float]],
            audio_input_path: str | None = None,
            tiling_config: TilingConfig | None = None,
            enhance_prompt: bool = False,
            output_path: str = '',
            video_chunks_number: int = 0,
            fps: int = 0,
            save_step_1_preview: bool = True,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor | None]:
        print("Start Call")
        startAt = time.time()
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16
        
        # --- LOAD AUDIO ---
        audio_waveform = None
        audio_latents = None
        
        if audio_input_path:
             print(f"Loading audio from {audio_input_path}")
             audio_waveform = load_audio_input(audio_input_path, AUDIO_SAMPLE_RATE, self.device)
             print("Encoding audio latents...")
             audio_latents = self.encode_audio_latents(audio_waveform)
             pass

        print("starting text encoder", time.time() - startAt)

        # --- DISK CACHE LOGIC START ---
        CACHE_DIR = "./prompt_embeddings_cache"
        os.makedirs(CACHE_DIR, exist_ok=True)

        image_identifier = images[0][0] if (len(images) > 0 and enhance_prompt) else "no_img"

        hash_input_str = (
            f"prompt:{prompt}|"
            f"neg:{negative_prompt}|"
            f"enhance:{enhance_prompt}|"
            f"seed:{seed if enhance_prompt else 'ignored'}|"
            f"img:{image_identifier}"
        )

        cache_filename = hashlib.md5(hash_input_str.encode('utf-8')).hexdigest() + ".pt"
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        context_p = None
        context_n = None

        if os.path.exists(cache_path):
            print(f"Disk cache hit! Loading embeddings from {cache_path}")
            try:
                cached_data = torch.load(cache_path, map_location=self.device)
                context_p, context_n = cached_data
            except Exception as e:
                print(f"Failed to load cache (corrupted?): {e}. Regenerating.")

        if context_p is None:
            print("Disk cache miss. Running text encoder.")
            text_encoder = self.stage_1_model_ledger.text_encoder()

            current_prompt = prompt
            if enhance_prompt:
                current_prompt = generate_enhanced_prompt(
                    text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
                )

            context_p, context_n = encode_text(text_encoder, prompts=[current_prompt, negative_prompt])

            print(f"Saving embeddings to {cache_path}")
            torch.save((context_p, context_n), cache_path)

            torch.cuda.synchronize()
            del text_encoder
            cleanup_memory()
        # --- DISK CACHE LOGIC END ---

        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n
        print("end text encoder", time.time() - startAt)

        print("Stage 1: Initial low resolution video generation.", time.time() - startAt)
        
        video_encoder = self.stage_1_model_ledger.video_encoder()
        transformer = self.stage_1_model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )

        if audio_latents is not None:
             expected_audio_shape = AudioLatentShape.from_video_pixel_shape(stage_1_output_shape, sample_rate=AUDIO_SAMPLE_RATE)
             target_frames = expected_audio_shape.frames
             current_frames = audio_latents.shape[2]
             
             if current_frames > target_frames:
                 print(f"Aligning audio: Trimming from {current_frames} to {target_frames}")
                 audio_latents = audio_latents[:, :, :target_frames, :]
             elif current_frames < target_frames:
                 print(f"Aligning audio: Padding from {current_frames} to {target_frames}")
                 pad_amount = target_frames - current_frames
                 audio_latents = torch.nn.functional.pad(audio_latents, (0, 0, 0, pad_amount))
        
        loop_audio_latents = None
        if audio_latents is not None:
             audio_state, audio_tools = noise_audio_state(
                 stage_1_output_shape,
                 noiser,
                 [], 
                 self.pipeline_components,
                 self.dtype,
                 self.device,
                 noise_scale=1.0,
                 initial_latent=audio_latents,
             )

             loop_audio_latents = einops.rearrange(audio_latents, "b c t f -> b t (c f)")
             in_features = transformer.velocity_model.audio_patchify_proj.in_features
             if loop_audio_latents.shape[-1] != in_features:
                 print(f"Aligning audio features for loop: {loop_audio_latents.shape[-1]} -> {in_features}")
                 loop_audio_latents = loop_audio_latents[..., :in_features]
                 
        def first_stage_denoising_loop(
                sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol, is_conditioning: bool = True
        ) -> tuple[LatentState, LatentState]:
            
            # Use masking explicitly since it's the non-distilled model
            if audio_state is not None and getattr(audio_state, 'denoise_mask', None) is not None:
                audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))

            v_x = video_state
            a_x = audio_state
            
            for i in range(len(sigmas) - 1):
                sigma_hat = sigmas[i]
                sigma_next = sigmas[i + 1]
                
                if loop_audio_latents is not None:
                     a_x = replace(a_x, latent=loop_audio_latents)

                denoised_v, denoised_a = multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,
                )(v_x, a_x, sigmas, i)

                d_v = (v_x.latent - denoised_v) / sigma_hat
                dt = sigma_next - sigma_hat
                v_x = replace(v_x, latent=v_x.latent + d_v * dt)

            return v_x, a_x

        stage_1_conditionings = []
        if images:
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
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_audio_latent=audio_latents if audio_latents is not None else None
        )
        print("Stage 1: End denoising loop.", time.time() - startAt)
        
        del transformer
        cleanup_memory()

        if save_step_1_preview:
            video_decoder = self.stage_1_model_ledger.video_decoder()
            decoded_video = vae_decode_video(video_state.latent, video_decoder, tiling_config, generator)
            del video_decoder
            cleanup_memory()
            
            decoded_audio = None
            if audio_latents is not None:
                vocoder = self.stage_1_model_ledger.vocoder()
                decoded_audio = vae_decode_audio(
                    audio_state.latent, self.stage_1_model_ledger.audio_decoder(), vocoder
                )
                del vocoder
                cleanup_memory()
            
            encode_video(
                video=decoded_video,
                fps=fps,
                audio=audio_waveform.cpu() if audio_waveform is not None else decoded_audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path.replace('.mp4', '_.mp4'),
                video_chunks_number=video_chunks_number,
            )

        print("Stage 2: Upsample and refine the video at higher resolution with distilled Checkpoint.", time.time() - startAt)
        
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_1_model_ledger.spatial_upsampler(),
        )
        print("Stage 2: Upsample and refine the video end.", time.time() - startAt)

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = []
        if images:
            stage_2_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )

        del video_encoder
        cleanup_memory()

        transformer = self.stage_2_model_ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
                sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol, is_conditioning: bool = True
        ) -> tuple[LatentState, LatentState]:
            
            # In Stage 2 (distilled), we use flow-matching interpolation so we don't break the joint trajectory
            if audio_state is not None and getattr(audio_state, 'denoise_mask', None) is not None:
                audio_state = replace(audio_state, denoise_mask=torch.ones_like(audio_state.denoise_mask))

            audio_noise = audio_state.latent.clone() if audio_state is not None else None
            
            v_x = video_state
            a_x = audio_state
            
            for i in range(len(sigmas) - 1):
                sigma_hat = sigmas[i]
                sigma_next = sigmas[i + 1]
                
                if loop_audio_latents is not None:
                     sigma_val = sigma_hat.item() if isinstance(sigma_hat, torch.Tensor) else sigma_hat
                     noised_audio = loop_audio_latents * (1.0 - sigma_val) + audio_noise * sigma_val
                     a_x = replace(a_x, latent=noised_audio.to(a_x.latent.dtype))
                     
                denoised_v, denoised_a = simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer, 
                )(v_x, a_x, sigmas, i)

                d_v = (v_x.latent - denoised_v) / sigma_hat
                dt = sigma_next - sigma_hat
                v_x = replace(v_x, latent=v_x.latent + d_v * dt)

            return v_x, a_x

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_latents,
        )
        print("Stage 2: Denoising loop end.", time.time() - startAt)
        
        del transformer
        cleanup_memory()

        decoded_video = vae_decode_video(video_state.latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator)
        decoded_audio = None
        if audio_latents is not None:
            decoded_audio = vae_decode_audio(
                audio_state.latent, self.stage_2_model_ledger.audio_decoder(), self.stage_2_model_ledger.vocoder()
            )
        print("Stage 2:vae decode video end.", time.time() - startAt)
        return decoded_video, audio_waveform.cpu() if audio_waveform is not None else decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_music_arg_parser()
    args = parser.parse_args()
    
    pipeline = MusicToVideoTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        stage_2_checkpoint_path=args.stage_2_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        quantization=args.quantization,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        audio_input_path=args.audio_input_path,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
        fps=args.frame_rate,
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
