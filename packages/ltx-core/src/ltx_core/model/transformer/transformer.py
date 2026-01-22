from dataclasses import dataclass, replace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
            self,
            idx: int,
            video: TransformerConfig | None = None,
            audio: TransformerConfig | None = None,
            rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
            norm_eps: float = 1e-6,
            attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(6, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(6, audio.dim))

        if audio is not None and video is not None:
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
            self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice, is_conditioning: bool = False
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        if not is_conditioning:
            if timestep.dim() > 2 and timestep.shape[1] > 1:
                timestep = timestep[:, [0], ...]

        table_slice = scale_shift_table[indices]
        if table_slice.device != timestep.device or table_slice.dtype != timestep.dtype:
            table_slice = table_slice.to(device=timestep.device, dtype=timestep.dtype)

        ada_values = (
                table_slice.unsqueeze(0).unsqueeze(0)
                + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
            self,
            scale_shift_table: torch.Tensor,
            batch_size: int,
            scale_shift_timestep: torch.Tensor,
            gate_timestep: torch.Tensor,
            num_scale_shift_values: int = 4,
            is_conditioning: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, slice(None, None), is_conditioning
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None), is_conditioning
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def forward(  # noqa: PLR0915
            self,
            video: TransformerArgs | None,
            audio: TransformerArgs | None,
            perturbations: BatchedPerturbationConfig | None = None,
            is_conditioning: bool = True
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        batch_size = video.x.shape[0] if video is not None else (audio.x.shape[0] if audio is not None else 0)
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        # --- Video Self-Attention & Cross-Attention ---
        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3), is_conditioning
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                # Optimization: Compute norm, then scale in-place to avoid allocating 'vx_scaled'
                norm_vx = rms_norm(vx, eps=self.norm_eps)
                norm_vx.mul_(1 + vscale_msa).add_(vshift_msa)

                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                attn_out = self.attn1(norm_vx, pe=video.positional_embeddings)
                vx = vx + attn_out * vgate_msa * v_mask

                del norm_vx, attn_out, v_mask

            # Optimization: Context Attention
            # We can reuse the norm calculation or just compute it fresh.
            # To be safe with residual connections, we compute fresh, but scale in-place.
            norm_vx = rms_norm(vx, eps=self.norm_eps)
            # No ada modulation for context attn in this block usually, but if we had it, we'd do it in-place here.
            attn_out = self.attn2(norm_vx, context=video.context, mask=video.context_mask)
            vx = vx + attn_out

            del norm_vx, attn_out, vshift_msa, vscale_msa, vgate_msa

        # --- Audio Self-Attention & Cross-Attention ---
        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3), is_conditioning
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = rms_norm(ax, eps=self.norm_eps)
                norm_ax.mul_(1 + ascale_msa).add_(ashift_msa)

                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                attn_out = self.audio_attn1(norm_ax, pe=audio.positional_embeddings)
                ax = ax + attn_out * agate_msa * a_mask

                del norm_ax, attn_out, a_mask

            del ashift_msa, ascale_msa, agate_msa

            # Audio Context Attention
            norm_ax = rms_norm(ax, eps=self.norm_eps)
            attn_out = self.audio_attn2(norm_ax, context=audio.context, mask=audio.context_mask)
            ax = ax + attn_out
            del norm_ax, attn_out

        # --- Audio - Video Cross Attention (MEMORY OPTIMIZED) ---
        if run_a2v or run_v2a:
            # These norms are allocated fresh.
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            # Helper to process A2V
            if run_a2v:
                (
                    scale_ca_audio_hidden_states_a2v,
                    shift_ca_audio_hidden_states_a2v,
                    _,
                    _,
                    _,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                )

                (
                    scale_ca_video_hidden_states_a2v,
                    shift_ca_video_hidden_states_a2v,
                    _,
                    _,
                    gate_out_a2v,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                )

                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)

                # OPTIMIZATION: If V2A is NOT running, we can modify vx_norm3/ax_norm3 in-place.
                # This prevents allocating 'vx_scaled' and 'ax_scaled' buffers.
                if not run_v2a:
                    vx_norm3.mul_(1 + scale_ca_video_hidden_states_a2v).add_(shift_ca_video_hidden_states_a2v)
                    ax_norm3.mul_(1 + scale_ca_audio_hidden_states_a2v).add_(shift_ca_audio_hidden_states_a2v)

                    attn_out = self.audio_to_video_attn(
                        vx_norm3,
                        context=ax_norm3,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings
                    )
                else:
                    # If V2A is running, we need the original norms for it, so we must allocate new scaled tensors.
                    vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
                    ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v

                    attn_out = self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings
                    )
                    del vx_scaled, ax_scaled

                vx = vx + attn_out * gate_out_a2v * a2v_mask

                del scale_ca_video_hidden_states_a2v, shift_ca_video_hidden_states_a2v
                del scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v
                del gate_out_a2v, a2v_mask, attn_out

            # Helper to process V2A
            if run_v2a:
                (
                    _,
                    _,
                    scale_ca_audio_hidden_states_v2a,
                    shift_ca_audio_hidden_states_v2a,
                    gate_out_v2a,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                )

                (
                    _,
                    _,
                    scale_ca_video_hidden_states_v2a,
                    shift_ca_video_hidden_states_v2a,
                    _,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                )

                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)

                # OPTIMIZATION: If A2V did NOT run, we can use the norms in-place.
                if not run_a2v:
                    ax_norm3.mul_(1 + scale_ca_audio_hidden_states_v2a).add_(shift_ca_audio_hidden_states_v2a)
                    vx_norm3.mul_(1 + scale_ca_video_hidden_states_v2a).add_(shift_ca_video_hidden_states_v2a)

                    attn_out = self.video_to_audio_attn(
                        ax_norm3,
                        context=vx_norm3,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings
                    )
                else:
                    # Both A2V and V2A ran. A2V preserved the norms (because of the `else` block above).
                    # So we still have the original norms here. We must allocate new.
                    ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                    vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a

                    attn_out = self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings
                    )
                    del ax_scaled, vx_scaled

                ax = ax + attn_out * gate_out_v2a * v2a_mask

                del scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a
                del scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a
                del gate_out_v2a, v2a_mask, attn_out

            del vx_norm3, ax_norm3

        # --- FFN Layers ---
        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None), is_conditioning
            )
            # Optimization: In-place scaling of the normalized input
            vx_scaled = rms_norm(vx, eps=self.norm_eps)
            vx_scaled.mul_(1 + vscale_mlp).add_(vshift_mlp)

            ffn_out = self.ff(vx_scaled)
            vx = vx + ffn_out * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled, ffn_out

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None), is_conditioning
            )
            # Optimization: In-place scaling
            ax_scaled = rms_norm(ax, eps=self.norm_eps)
            ax_scaled.mul_(1 + ascale_mlp).add_(ashift_mlp)

            ffn_out = self.audio_ff(ax_scaled)
            ax = ax + ffn_out * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled, ffn_out

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None
