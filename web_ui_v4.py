import gradio as gr
import subprocess
import os
import datetime
import uuid
import threading
import time
import sys
from collections import deque

# --- Configuration & Defaults ---
DEFAULT_CHECKPOINT = "./models/ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA = "./models/gemma3"
DEFAULT_UPSAMPLER = "./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
LORA_ROOT = "./models/loras"

# LoRA List
LORA_OPTIONS = [
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
    "1280x704 (Landscape)": {"w": 1280, "h": 704, "max_frames": 225},
    "704x1280 (Vertical)": {"w": 704, "h": 1280, "max_frames": 225},
    "1536x1024 (Standard)": {"w": 1536, "h": 1024, "max_frames": 121},
    "1024x1536 (Vertical)": {"w": 1024, "h": 1536, "max_frames": 121},
    "1600x896 (Landscape)": {"w": 1600, "h": 896, "max_frames": 145},
    "896x1600 (Vertical)": {"w": 896, "h": 1600, "max_frames": 145},
    "1920x1088 (HD)": {"w": 1920, "h": 1088, "max_frames": 97},
    "1088x1920 (HD Vert)": {"w": 1088, "h": 1920, "max_frames": 97},
    "2560x1408 (2K)": {"w": 2560, "h": 1408, "max_frames": 49},
    "1408x2560 (2K Vert)": {"w": 1408, "h": 2560, "max_frames": 49},
    "3840x2176 (4K)": {"w": 3840, "h": 2176, "max_frames": 17},
}

# Cinematic

# --- Prompt Construction Data ---
# Animation: stop-motion, 2D/3D animation, claymation, hand-drawn
# Stylized: comic book, cyberpunk, 8-bit pixel, surreal, minimalist, painterly, illustrated
# Cinematic: period drama, film noir, fantasy, epic space opera, thriller, modern romance, experimental film, arthouse, documentary
STYLES = ["Cinematic", "Photorealistic", "3D Animation", "Anime", "Vintage Film (VHS)", "Film Noir", "Cyberpunk",
          "Oil Painting", "Claymation"]
SHOT_TYPES = ["Wide establishing", "Medium", "Close-up", "Extreme close-up", "Over-the-shoulder", "Low angle",
              "High angle", "Overhead"]
LIGHTING = ["Natural sunlight", "Golden hour", "Cinematic", "Volumetric fog", "Neon glow", "Dark and moody",
            "Studio lighting", "Soft rim light"]
CAM_MOVES = ["static frame", "wide establishing shot", "over-the-shoulder", "handheld movement", "overhead view",
             "pushes in", "pulls back",
             "tilts upward", "circles around", "pans across", "follows", "tracks"]  # ok

# Establish the shot. Use cinematography terms that match your preferred film genre. Include aspects like scale or specific category characteristics to further refine the style you’re looking for.
# Set the scene. Describe lighting conditions, color palette, surface textures, and atmosphere to shape the mood.
# Describe the action. Write the core action as a natural sequence, flowing from beginning to end.
# Define your character(s). Include age, hairstyle, clothing, and distinguishing details. Express emotions through physical cues.
# Identify camera movement(s). Specify when the view should shift and how. Including how subjects or objects appear after the camera motion gives the model a better idea of how to finish the motion.‍
# Describe the audio. Use clear descriptions for ambient sounds, music, audio, and speech. For dialogue, place the text between quotation marks and (if required) mention the language and accent you would like the character to have.


# Keep your prompt in a single flowing paragraph to give the model a cohesive scene to work with.
# Use present tense verbs to describe movement and action.
# Match your detail to the shot scale. Closeups need more precise detail than wide shots.
# When describing camera movement, focus on the camera’s relationship to the subject.
# You should expect to write 4 to 8 descriptive sentences to cover all the key aspects of the prompt.
# Don’t be afraid to iterate! LTX-2 is designed for fast experimentation, so refining your prompt is part of the workflow.

# Scale indicators: expansive, epic, intimate, claustrophobic
# Film characteristics: jittery stop-motion, pixelated edges, lens flares, film grain
# Pacing and temporal effects: slow motion, time-lapse, rapid cuts, lingering shot, continuous shot, freeze-frame, fade-in, fade-out, seamless transition, dynamic movement, sudden stop
# Specific visual effects (if relevant): particle systems, motion blur, depth of field

# Lighting conditions: flickering candles, neon glow, natural sunlight, dramatic shadows
# Textures: rough stone, smooth metal, worn fabric, glossy surfaces
# Color palette: vibrant, muted, monochromatic, high contrast
# Atmospheric elements: fog, rain, dust, particles, smoke

# Sound and Voice
# Setting: Ambient coffeeshop noises, dripping rain and wind blowing, forest ambience with birds singing
# Dialogue style: Energetic announcer, resonant voice with gravitas, distorted radio-style, robotic monotone, childlike curiosity
# Volume: quiet whisper, mutters, shouts, screams


# ‍Cinematic close-up shot ‍cinematic lighting shallow depth of field, and natural motion.

# What Works Well with LTX-2
# ‍Cinematic compositions:
# ‍Wide, medium, and close-up shots with thoughtful lighting, shallow depth of field, and natural motion.
# Emotive human moments:
# ‍LTX-2 excels at single-subject emotional expressions, subtle gestures, and facial nuance.
# Atmosphere & setting:
# ‍Weather effects like fog, mist, golden hour light, soft shadows, rain, reflections, and ambient textures all help ground the scene.
# Clean, readable camera language:
# ‍Clear directions like “slow dolly in,” “handheld tracking,” or “over-the-shoulder” improve consistency.
# Stylized aesthetics:
# ‍Painterly, noir, analog film look, fashion editorial, pixelated animation, or surreal art styles work especially well when named early in the prompt.
# Lighting and mood control:
# ‍Backlighting, color palettes, soft rim light, flickering lamps — these anchor tone better than generic mood words.
# Voice:
# ‍Characters can talk and sing in various languages.

# --- Global Queue System ---
JOB_QUEUE = deque()
QUEUE_LOCK = threading.Lock()
CURRENT_LOG = "System Ready. Waiting for jobs..."
LATEST_VIDEO_PATH = None
IS_PROCESSING = False
CURRENT_JOB_ID = None
CURRENT_PROCESS = None
PREVIEW_VIDEO_PATH = None
CURRENT_OUTPUT_PATH = None


# --- Logic Functions ---

def build_ltx_prompt_text(style, shot, subject, env, light, cam):
    parts = []
    opener = ""
    if style: opener += f"{style} "
    if shot: opener += f"{shot} shot"
    if opener:
        parts.append(f"A {opener.strip()} of")
    else:
        parts.append("A shot of")

    parts.append(subject if subject else "a subject")

    loc_details = []
    if env: loc_details.append(f"in {env}")
    if light: loc_details.append(f"with {light}")
    if loc_details: parts.append(" ".join(loc_details))

    full_text = " ".join(parts) + "."
    if cam: full_text += f" The camera {cam.lower()}."
    return full_text


def get_preset_frames(preset_key, is_safe_mode, current_val):
    if not is_safe_mode: return current_val
    if preset_key in PRESETS: return PRESETS[preset_key]["max_frames"]
    return 121


# --- Worker Logic ---

def process_job_logic(job):
    """Internal function to run the actual generation logic"""
    global CURRENT_LOG, LATEST_VIDEO_PATH, CURRENT_PROCESS, PREVIEW_VIDEO_PATH, CURRENT_OUTPUT_PATH

    # Reset preview/process state
    PREVIEW_VIDEO_PATH = None
    CURRENT_PROCESS = None
    prompt = job['prompt']
    width = PRESETS[job['preset']]["w"]
    height = PRESETS[job['preset']]["h"]
    seed = job['seed']
    if job['randomize_seed']:
        seed = int(os.urandom(4).hex(), 16) % (2 ** 32)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{timestamp}_{job['id']}.mp4"
    output_path = os.path.abspath(output_filename)
    CURRENT_OUTPUT_PATH = output_path

    CURRENT_LOG += f"\n\n--- STARTED JOB: {job['id']} ---\nPrompt: {prompt}\nSeed: {seed}\n"

    # Build Command
    cmd = [
        sys.executable, "-m", "ltx_pipelines.distilled",
        # "kernprof", "-l", "-v", "-m", "ltx_pipelines.distilled",
        "--distilled-checkpoint-path", job['checkpoint_path'],
        "--gemma-root", job['gemma_path'],
        "--spatial-upsampler-path", job['upsampler_path'],
        "--prompt", prompt,
        "--output-path", output_path,
        "--width", str(width),
        "--height", str(height),
        "--num-frames", str(int(job['num_frames'])),
        "--frame-rate", str(job['frame_rate']),
        "--num-inference-steps", str(int(job['steps'])),
        "--seed", str(int(seed)),
        # "--enable-chunked-stage2"
        "--quantization", "fp8-cast"
    ]

    if job['enhance_prompt']: cmd.append("--enhance-prompt")
    if job['disable_audio']: cmd.append("--disable-audio")

    # Images
    for path, idx, strength in job['images']:
        if path is not None:
            latent_idx = int(idx) // 8
            cmd.extend(["--image", path, str(latent_idx), str(float(strength))])

    # LoRAs
    for lora_name in job['loras']:
        lora_full_path = os.path.join(LORA_ROOT, f"{lora_name.lower()}.safetensors")
        cmd.extend(["--lora", lora_full_path, "1.0"])

    import shlex
    full_command_str = " ".join(shlex.quote(arg) for arg in cmd)
    CURRENT_LOG += f"Command:\n{full_command_str}\n\n--- OUTPUT LOG ---\n"

    # Execution
    try:
        CURRENT_PROCESS = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True
        )

        for line in CURRENT_PROCESS.stdout:
            CURRENT_LOG += line

        CURRENT_PROCESS.wait()

        if CURRENT_PROCESS.returncode == 0 and os.path.exists(output_path):
            CURRENT_LOG += f"\n--- JOB COMPLETE ---\nSaved: {output_path}\n"
            LATEST_VIDEO_PATH = output_path
        else:
            CURRENT_LOG += f"\n--- JOB FAILED ---\nReturn Code: {CURRENT_PROCESS.returncode}\n"
    except Exception as e:
        CURRENT_LOG += f"\n--- EXCEPTION ---\n{str(e)}\n"
    finally:
        CURRENT_PROCESS = None


def cancel_job():
    global CURRENT_PROCESS, CURRENT_LOG
    if CURRENT_PROCESS:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(CURRENT_PROCESS.pid)], capture_output=True)
            CURRENT_LOG += "\n--- JOB CANCELLATION REQUESTED AND EXECUTED ---\n"
            return "Cancellation requested and executed via taskkill."
        except Exception as e:
            try:
                CURRENT_PROCESS.terminate()
                return f"Taskkill failed, tried terminate: {str(e)}"
            except:
                return f"Error cancelling: {str(e)}"
    return "No active process to cancel."


def worker_thread():
    """Background thread that constantly checks for jobs"""
    global IS_PROCESSING, CURRENT_JOB_ID, CURRENT_LOG
    while True:
        job = None
        with QUEUE_LOCK:
            if len(JOB_QUEUE) > 0:
                job = JOB_QUEUE.popleft()
                IS_PROCESSING = True
                CURRENT_JOB_ID = job['id']
            else:
                IS_PROCESSING = False
                CURRENT_JOB_ID = None

        if job:
            process_job_logic(job)
        else:
            time.sleep(1)


# Start Worker
threading.Thread(target=worker_thread, daemon=True).start()


# --- UI Functions ---

def enqueue_job(
        prompt, preset, num_frames, disable_audio, frame_rate, steps, seed, randomize_seed, enhance_prompt, enable_fp8,
        checkpoint_path, gemma_path, upsampler_path,
        img1_path, img1_idx, img1_str,
        img2_path, img2_idx, img2_str,
        img3_path, img3_idx, img3_str,
        selected_loras
):
    job_id = str(uuid.uuid4())[:4]

    job_data = {
        "id": job_id,
        "prompt": prompt,
        "preset": preset,
        "num_frames": num_frames,
        "disable_audio": disable_audio,
        "frame_rate": frame_rate,
        "steps": steps,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "enhance_prompt": enhance_prompt,
        "enable_fp8": enable_fp8,
        "checkpoint_path": checkpoint_path,
        "gemma_path": gemma_path,
        "upsampler_path": upsampler_path,
        "images": [
            (img1_path, img1_idx, img1_str),
            (img2_path, img2_idx, img2_str),
            (img3_path, img3_idx, img3_str)
        ],
        "loras": selected_loras
    }

    with QUEUE_LOCK:
        JOB_QUEUE.append(job_data)
        q_pos = len(JOB_QUEUE)

    return f"Job {job_id} queued. Position: {q_pos}"


def update_monitor():
    """Polled by the UI to get latest logs and queue status"""
    global PREVIEW_VIDEO_PATH

    # Check for intermediate preview file
    if CURRENT_OUTPUT_PATH and PREVIEW_VIDEO_PATH is None:
        preview_file = CURRENT_OUTPUT_PATH.replace('.mp4', '_.mp4')
        if os.path.exists(preview_file):
            PREVIEW_VIDEO_PATH = preview_file

    q_len = len(JOB_QUEUE)
    status_str = f"Queue Size: {q_len}"
    if IS_PROCESSING:
        status_str += f" | Processing Job: {CURRENT_JOB_ID}"
    else:
        status_str += " | Idle"

    return LATEST_VIDEO_PATH, PREVIEW_VIDEO_PATH, CURRENT_LOG, status_str


# --- UI Theme & Layout ---

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
#status_box { font-weight: bold; color: #475569; }
"""

with gr.Blocks(title="LTX-2 Studio + Queue", theme=theme, css=css) as demo:
    gr.Markdown("## 🎬 LTX-2 Distilled Web Interface (Queue Enabled)")

    with gr.Row():
        # Left Column: Controls
        with gr.Column(scale=3):
            # --- Prompt Constructor ---
            with gr.Accordion("📝 Prompt Constructor (LTX Guide Based)", open=False):
                with gr.Row():
                    pc_style = gr.Dropdown(choices=STYLES, label="Style", value="Cinematic")
                    pc_shot = gr.Dropdown(choices=SHOT_TYPES, label="Shot Type", value="Medium")
                    pc_light = gr.Dropdown(choices=LIGHTING, label="Lighting", value="Golden hour")

                pc_subject = gr.Textbox(label="Subject & Action", placeholder="e.g., a futuristic robot walking...",
                                        lines=2)
                pc_env = gr.Textbox(label="Environment", placeholder="e.g., a dusty desert")

                with gr.Row():
                    pc_cam = gr.Dropdown(choices=CAM_MOVES, label="Camera Movement", value="static frame")
                    pc_build_btn = gr.Button("⬇️ Insert into Prompt", variant="secondary")

            # Main Prompt
            prompt = gr.Textbox(label="Final Prompt", placeholder="Describe your video scene here...", lines=3)

            with gr.Row():
                with gr.Column(scale=1):
                    preset = gr.Dropdown(label="Resolution", choices=list(PRESETS.keys()), value="1536x1024 (Standard)")
                    safe_mode = gr.Checkbox(label="8GB Safe Mode", value=True)
                    disable_audio = gr.Checkbox(label="Disable audio", value=False)

                with gr.Column(scale=1):
                    num_frames = gr.Slider(label="Frames", minimum=9, maximum=257, step=8, value=121)
                    fps = gr.Slider(label="FPS", minimum=8, maximum=60, step=1, value=24)

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=8)
                    seed = gr.Number(label="Seed", value=10, precision=0)
                with gr.Row():
                    random_seed = gr.Checkbox(label="Random Seed", value=True)
                    enable_fp8 = gr.Checkbox(label="FP8", value=True)
                    enhance_prompt = gr.Checkbox(label="Enhance", value=False)

                checkpoint_path = gr.Textbox(label="Checkpoint", value=DEFAULT_CHECKPOINT)
                gemma_path = gr.Textbox(label="Gemma Root", value=DEFAULT_GEMMA)
                upsampler_path = gr.Textbox(label="Upsampler", value=DEFAULT_UPSAMPLER)

        # Right Column: Output & Monitor
        with gr.Column(scale=4):
            # Status Bar
            queue_status = gr.Textbox(label="System Status", value="Idle", interactive=False, elem_id="status_box")

            # Video Output
            out_video = gr.Video(label="Last Completed Video", height=400, autoplay=True)

            # Preview Video
            preview_video = gr.Video(label="Stage 1 Preview", height=300, autoplay=True)

            # Button (Adds to Queue)
            with gr.Row():
                generate_btn = gr.Button("➕ Add to Queue", variant="primary", size="lg")
                cancel_btn = gr.Button("🛑 Cancel Current Job", variant="secondary", size="lg")

            add_result_msg = gr.Markdown("")  # Feedback for button click

            with gr.Accordion("Console Log", open=True):
                console_log = gr.Textbox(label="Worker Log", lines=10, max_lines=20, interactive=False,
                                         elem_id="console_log")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎨 LoRA Adapters")
            lora_checks = gr.CheckboxGroup(choices=LORA_OPTIONS, label=None)

        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Image Conditioning")
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

    # --- Timers & Events ---

    timer = gr.Timer(1)
    timer.tick(
        fn=update_monitor,
        inputs=None,
        outputs=[out_video, preview_video, console_log, queue_status]
    )

    # 2. Add to Queue Action
    generate_btn.click(
        fn=enqueue_job,
        inputs=[
            prompt, preset, num_frames, disable_audio, fps, steps, seed, random_seed, enhance_prompt, enable_fp8,
            checkpoint_path, gemma_path, upsampler_path,
            i1_img, i1_idx, i1_str,
            i2_img, i2_idx, i2_str,
            i3_img, i3_idx, i3_str,
            lora_checks
        ],
        outputs=[add_result_msg]
    )

    # 2b. Cancel Job Action
    cancel_btn.click(
        fn=cancel_job,
        inputs=None,
        outputs=[add_result_msg]
    )

    # 3. Prompt Construction
    pc_build_btn.click(
        fn=build_ltx_prompt_text,
        inputs=[pc_style, pc_shot, pc_subject, pc_env, pc_light, pc_cam],
        outputs=prompt
    )

    # 4. Presets Logic
    preset.change(fn=get_preset_frames, inputs=[preset, safe_mode, num_frames], outputs=num_frames)
    safe_mode.change(fn=get_preset_frames, inputs=[preset, safe_mode, num_frames], outputs=num_frames)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
