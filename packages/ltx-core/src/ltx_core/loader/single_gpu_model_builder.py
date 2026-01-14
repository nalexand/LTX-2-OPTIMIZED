import logging
from dataclasses import dataclass, field, replace
from typing import Generic

import torch

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)

from loguru import logger
from accelerate import dispatch_model, infer_auto_device_map


@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU or offloaded via Accelerate.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)

    def lora(self, lora_path: str, strength: float = 1.0, sd_ops: SDOps | None = None) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def model_config(self) -> dict:
        first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
        return self.model_loader.metadata(first_shard_path)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        with torch.device("meta"):
            model = self.model_class_configurator.from_config(config)
        for module_op in module_ops:
            if module_op.matcher(model):
                model = module_op.mutator(model)
        return model

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        state_dict = registry.get(paths, sd_ops)
        if state_dict is None:
            state_dict = self.model_loader.load(paths, sd_ops=sd_ops, device=device)
            registry.add(paths, sd_ops=sd_ops, state_dict=state_dict)
        return state_dict

    def _return_model(self, meta_model: ModelType, device: torch.device) -> ModelType:
        uninitialized_params = [name for name, param in meta_model.named_parameters() if str(param.device) == "meta"]
        uninitialized_buffers = [name for name, buffer in meta_model.named_buffers() if str(buffer.device) == "meta"]
        if uninitialized_params or uninitialized_buffers:
            logger.warning(f"Uninitialized parameters or buffers: {uninitialized_params + uninitialized_buffers}")
            return meta_model
        retval = meta_model.to(device)
        return retval

    def build(
            self,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            max_memory: dict[int | str, str] | None = None
    ) -> ModelType:
        target_device = torch.device("cuda") if device is None else device

        # 1. Get Config and Meta Model
        config = self.model_config()
        meta_model = self.meta_model(config, self.module_ops)

        # 2. Load Base State Dict
        model_paths = self.model_path if isinstance(self.model_path, tuple) else [self.model_path]
        load_device = target_device if max_memory is None else torch.device("cpu")
        model_state_dict = self.load_sd(model_paths, sd_ops=self.model_sd_ops, registry=self.registry,
                                        device=load_device)

        # 3. Handle LoRAs
        lora_strengths = [lora.strength for lora in self.loras]
        final_sd_map = {}

        if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
            final_sd_map = model_state_dict.sd
        else:
            # Convert LoRAs to float32 on CPU to prevent slow BF16 emulation
            lora_state_dicts = []
            for lora in self.loras:
                lsd = self.load_sd([lora.path], sd_ops=lora.sd_ops, registry=self.registry, device=load_device)

                if load_device.type == "cpu":
                    # In-place conversion of LoRA tensors to float32 for speed
                    # This speeds up the matmul in apply_loras significantly
                    for k, v in lsd.sd.items():
                        if v.dtype in [torch.bfloat16, torch.float16]:
                            lsd.sd[k] = v.to(dtype=torch.float32)

                lora_state_dicts.append(lsd)

            lora_sd_and_strengths = [
                LoraStateDictWithStrength(sd, strength)
                for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
            ]

            dest_sd = model_state_dict if isinstance(self.registry, DummyRegistry) else None

            final_sd_obj = apply_loras(
                model_sd=model_state_dict,
                lora_sd_and_strengths=lora_sd_and_strengths,
                dtype=dtype,
                destination_sd=dest_sd,
            )
            final_sd_map = final_sd_obj.sd

        # 4. Cast Dtypes if requested
        if dtype is not None:
            final_sd_map = {k: v.to(dtype=dtype) for k, v in final_sd_map.items()}

        # 5. Load State Dict into Model
        meta_model.load_state_dict(final_sd_map, strict=False, assign=True)

        # 6. Return based on Offloading strategy
        if max_memory is not None:
            logger.info(f"Dispatching model with max_memory constraints: {max_memory}")
            no_split_modules = getattr(self.model_class_configurator, "no_split_modules", None)
            device_map = infer_auto_device_map(
                meta_model,
                max_memory=max_memory,
                no_split_module_classes=no_split_modules,
                dtype=dtype
            )
            model = dispatch_model(meta_model, device_map=device_map)
            return model

        return self._return_model(meta_model, target_device)
