import torch


def fused_add_round_kernel(
        x: torch.Tensor,
        output: torch.Tensor,
        seed: int,
        n_elements: int,  # Kept for signature compatibility, but unused
        EXPONENT_BIAS: int,
        MANTISSA_BITS: int,
        BLOCK_SIZE: int = None,  # Kept for signature compatibility, but unused
):
    """
    Native PyTorch implementation of the fused_add_round_kernel.

    This performs:
    1. Upcast 8-bit weights (x) to match output precision.
    2. Add output weights (deltas) to x.
    3. Calculate the epsilon (quantization noise step) based on the target
       Float8 parameters (EXPONENT_BIAS, MANTISSA_BITS).
    4. Apply stochastic rounding (add noise proportional to epsilon).
    5. Store back to output.
    """

    # 1. Setup Generators for stochastic rounding
    # We use a specific generator to respect the seed argument
    gen = torch.Generator(device=output.device).manual_seed(seed)

    # 2. Load and Cast to calculation precision (Float32 for safety, or Float16)
    # Using Float32 ensures high precision during the intermediate math
    val_x = x.to(torch.float32)
    val_delta = output.to(torch.float32)

    # x = x + delta
    val = val_x + val_delta

    # 3. Calculate Epsilon (The Stochastic Rounding Step)
    # The Triton kernel calculates epsilon based on the magnitude of 'val'
    # mapped onto the specific Float8 exponent grid.

    # Extract exponent: val = mantissa * 2^exp.
    # torch.frexp returns exp such that 0.5 <= |mantissa| < 1.0.
    # IEEE 754 log2(x) is (exp - 1).
    _, exp_obj = torch.frexp(val)
    unbiased_exp = exp_obj - 1

    # Map to target Float8 exponent space
    target_exp = unbiased_exp + EXPONENT_BIAS

    # Clamp exponent to target dtype range.
    # Max is standard formulation (2*Bias + 1).
    # Min is 1. Why 1? In the original Triton kernel, subnormals (exp <= 0)
    # utilize a constant epsilon calculated based on exponent=1 (the smallest normal).
    max_exponent = 2 * EXPONENT_BIAS + 1
    target_exp_clamped = torch.clamp(target_exp, min=1, max=max_exponent)

    # Calculate ULP exponent: E_target - BIAS - Mantissa_Bits
    eps_exponent = target_exp_clamped - EXPONENT_BIAS - MANTISSA_BITS

    # Convert exponent to actual epsilon value: 2^eps_exponent
    eps = torch.pow(2.0, eps_exponent.to(torch.float32))

    # Mask epsilon where value is exactly 0 (matches `tl.where(x == 0, 0.0, eps)`)
    eps = torch.where(val == 0, 0.0, eps)

    # 4. Generate Random Noise [-0.5, 0.5]
    rand_vals = torch.rand(val.shape, generator=gen, device=val.device) - 0.5

    # 5. Apply Stochastic Rounding
    # output = x + (noise * epsilon)
    result = val + (rand_vals * eps)

    # 6. Store Result
    # In-place update of the output tensor, cast to bfloat16
    output.copy_(result.to(torch.bfloat16))

    # No return value needed as operation is in-place on output_ptr/output