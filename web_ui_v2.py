import gradio as gr
import subprocess
import os
import datetime
import uuid
import sys
import shlex

# --- Configuration & Defaults ---
DEFAULT_CHECKPOINT = "./models/ltx-2-19b-distilled-fp8.safetensors"
DEFAULT_GEMMA = "./models/gemma3"
DEFAULT_UPSAMPLER = "./models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
LORA_ROOT = "./models/loras"

# LoRA List
LORA_OPTIONS = [
    #"LTX-2-19b-IC-LoRA-Canny-Control",
    #"LTX-2-19b-IC-LoRA-Depth-Control",
    #"LTX-2-19b-IC-LoRA-Detailer",
    #"LTX-2-19b-IC-LoRA-Pose-Control",
    "LTX-2-19b-LoRA-Camera-Control-Dolly-In",
    "LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
    "LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
    "LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
    "LTX-2-19b-LoRA-Camera-Control-Jib-Down",
    "LTX-2-19b-LoRA-Camera-Control-Jib-Up",
    "LTX-2-19b-LoRA-Camera-Control-Static"
]

# Resolution Presets with Max Frame Data for 8GB VRAM
PRESETS = {
    "1280x704 (Landscape)": {"w": 1280, "h": 704, "max_frames": 209},
    "704x1280 (Vertical)": {"w": 704, "h": 1280, "max_frames": 209},

    "1536x1024 (Standard)": {"w": 1536, "h": 1024, "max_frames": 121},
    "1024x1536 (Vertical)": {"w": 1024, "h": 1536, "max_frames": 121},

    "1600x896 (Landscape)": {"w": 1600, "h": 896, "max_frames": 121},
    "896x1600 (Vertical)": {"w": 8996, "h": 1600, "max_frames": 121},

    "1920x1088 (HD)": {"w": 1920, "h": 1088, "max_frames": 89},
    "1088x1920 (HD Vert)": {"w": 1088, "h": 1920, "max_frames": 89},

    "2560x1408 (2K)": {"w": 2560, "h": 1408, "max_frames": 49},
    "1408x2560 (2K Vert)": {"w": 1408, "h": 2560, "max_frames": 49},

    "3840x2176 (4K)": {"w": 3840, "h": 2176, "max_frames": 17},
}


# --- Logic Functions ---

def get_preset_frames(preset_key, is_safe_mode, current_val):
    """Updates frame count slider based on preset and safe mode"""
    if not is_safe_mode:
        return current_val  # Do not change if safe mode is off

    if preset_key in PRESETS:
        return PRESETS[preset_key]["max_frames"]
    return 121


def run_generation(
        prompt,
        resolution_preset,
        num_frames,
        frame_rate,
        steps,
        seed,
        randomize_seed,
        enhance_prompt,
        enable_fp8,
        # Paths
        checkpoint_path,
        gemma_path,
        upsampler_path,
        # Images
        img1_path, img1_idx, img1_str,
        img2_path, img2_idx, img2_str,
        img3_path, img3_idx, img3_str,
        # LoRAs
        selected_loras
):
    # 1. Setup Data
    width = PRESETS[resolution_preset]["w"]
    height = PRESETS[resolution_preset]["h"]

    if randomize_seed:
        seed = int(os.urandom(4).hex(), 16) % (2 ** 32)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"output_{timestamp}_{unique_id}.mp4"
    output_path = os.path.abspath(output_filename)

    # 2. Build Command
    cmd = [
        sys.executable, "-m", "ltx_pipelines.distilled",
        "--checkpoint-path", checkpoint_path,
        "--gemma-root", gemma_path,
        "--spatial-upsampler-path", upsampler_path,
        "--prompt", prompt,
        "--output-path", output_path,
        "--width", str(width),
        "--height", str(height),
        "--num-frames", str(int(num_frames)),
        "--frame-rate", str(frame_rate),
        "--num-inference-steps", str(int(steps)),
        "--seed", str(int(seed))
    ]

    if enable_fp8:
        cmd.append("--enable-fp8")
    if enhance_prompt:
        cmd.append("--enhance-prompt")

    # Images
    images = [
        (img1_path, img1_idx, img1_str),
        (img2_path, img2_idx, img2_str),
        (img3_path, img3_idx, img3_str)
    ]
    for path, idx, strength in images:
        if path is not None:
            latent_idx = int(idx) // 8
            cmd.extend(["--image", path, str(latent_idx), str(float(strength))])

    # LoRAs
    for lora_name in selected_loras:
        lora_full_path = os.path.join(LORA_ROOT, f"{lora_name.lower()}.safetensors")
        cmd.extend(["--lora", lora_full_path, "1.0"])

    # 3. Execution with Real-time Logging
    full_command_str = " ".join(shlex.quote(arg) for arg in cmd)
    log_buffer = f"Command:\n{full_command_str}\n\n--- OUTPUT LOG ---\n"
    yield None, log_buffer  # Clear video, show start log

    try:
        # Popen allows reading stdout line by line
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            log_buffer += line
            yield None, log_buffer  # Stream logs to UI

        process.wait()

        if process.returncode == 0 and os.path.exists(output_path):
            log_buffer += f"\n\nSUCCESS: Video saved to {output_path}"
            yield output_path, log_buffer
        else:
            log_buffer += f"\n\nERROR: Process failed or output file missing."
            yield None, log_buffer

    except Exception as e:
        log_buffer += f"\n\nEXCEPTION: {str(e)}"
        yield None, log_buffer


# --- UI Theme & Layout ---

# Custom neutral theme (slate/gray)
theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
).set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
    button_primary_background_fill="*primary_600",
    button_primary_text_color="white",
)

css = """
.gradio-container { max_width: 1400px !important; }
textarea { font-family: monospace; }
"""

with gr.Blocks(title="LTX-2 Studio", theme=theme, css=css) as demo:
    gr.Markdown("## 🎬 LTX-2 Distilled Web Interface")

    with gr.Row():
        # Left Column: Controls
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", placeholder="Describe your video scene here...", lines=3)

            with gr.Row():
                with gr.Column(scale=1):
                    # Resolution & Safe Mode
                    preset = gr.Dropdown(
                        label="Resolution",
                        choices=list(PRESETS.keys()),
                        value="1536x1024 (Standard)"
                    )
                    safe_mode = gr.Checkbox(
                        label="8GB VRAM Safe Mode",
                        value=True,
                        info="Auto-limits max frames to prevent OOM"
                    )

                with gr.Column(scale=1):
                    # Generation Params
                    num_frames = gr.Slider(label="Number of Frames", minimum=9, maximum=257, step=8, value=121)
                    fps = gr.Slider(label="Frame Rate", minimum=8, maximum=60, step=1, value=24)

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    steps = gr.Slider(label="Inference Steps", minimum=8, maximum=8, step=1, value=8, info="Fixed to 8 for distilled model")
                    seed = gr.Number(label="Seed", value=10, precision=0)

                with gr.Row():
                    random_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    enable_fp8 = gr.Checkbox(label="Enable FP8 (Required for 8GB vram)", value=True)
                    enhance_prompt = gr.Checkbox(label="Enhance Prompt (slow +1..3 min)", value=False)

                gr.Markdown("### Model Paths")
                checkpoint_path = gr.Textbox(label="Checkpoint", value=DEFAULT_CHECKPOINT)
                gemma_path = gr.Textbox(label="Gemma Root", value=DEFAULT_GEMMA)
                upsampler_path = gr.Textbox(label="Upsampler", value=DEFAULT_UPSAMPLER)

        # Right Column: Output & Media
        with gr.Column(scale=4):
            out_video = gr.Video(label="Generated Result", height=400)
            generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")

            with gr.Accordion("Console Log", open=True):
                console_log = gr.Textbox(label="Terminal Output", lines=10, max_lines=20, interactive=False,
                                         elem_id="console_log")

    gr.Markdown("---")

    with gr.Row():
        # LoRAs Column
        with gr.Column(scale=1):
            gr.Markdown("### 🎨 LoRA Adapters")
            lora_checks = gr.CheckboxGroup(
                choices=LORA_OPTIONS,
                label=None,
                info="Applied with strength 1.0"
            )

        # Image Conditioning Column
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Image Conditioning (Optional)")
            with gr.Row():
                # Image 1
                with gr.Group():
                    i1_img = gr.Image(type="filepath", label="Ref Image 1", height=150)
                    i1_idx = gr.Number(label="Frame Index", value=0)
                    i1_str = gr.Slider(label="Strength", minimum=0, maximum=1, value=0.8)

                # Image 2
                with gr.Group():
                    i2_img = gr.Image(type="filepath", label="Ref Image 2", height=150)
                    i2_idx = gr.Number(label="Frame Index", value=0)
                    i2_str = gr.Slider(label="Strength", minimum=0, maximum=1, value=0.8)

                # Image 3
                with gr.Group():
                    i3_img = gr.Image(type="filepath", label="Ref Image 3", height=150)
                    i3_idx = gr.Number(label="Frame Index", value=0)
                    i3_str = gr.Slider(label="Strength", minimum=0, maximum=1, value=0.8)

    # --- Event Wiring ---

    # Logic: When Preset OR Safe Mode changes, update the Num Frames Slider
    preset.change(
        fn=get_preset_frames,
        inputs=[preset, safe_mode, num_frames],
        outputs=num_frames
    )

    safe_mode.change(
        fn=get_preset_frames,
        inputs=[preset, safe_mode, num_frames],
        outputs=num_frames
    )

    # Logic: Run Generation
    generate_btn.click(
        fn=run_generation,
        inputs=[
            prompt, preset, num_frames, fps, steps, seed, random_seed, enhance_prompt, enable_fp8,
            checkpoint_path, gemma_path, upsampler_path,
            i1_img, i1_idx, i1_str,
            i2_img, i2_idx, i2_str,
            i3_img, i3_idx, i3_str,
            lora_checks
        ],
        outputs=[out_video, console_log]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)