import gradio as gr
import subprocess
import os
import datetime
import threading
import sys
import math
import torchaudio
from collections import deque
import cv2

# --- Configuration & Defaults ---
DEFAULT_CHECKPOINT = "./models/ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA = "./models/gemma3"
DEFAULT_UPSAMPLER = "./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
AUDIO_CLIPS_DIR = "./audio_clips"

# --- Global State ---
JOB_QUEUE = deque()
QUEUE_LOCK = threading.Lock()
CURRENT_LOG = "System Ready. Waiting for jobs..."
LATEST_VIDEO_PATH = None
IS_PROCESSING = False
STOP_GENERATION = False
CURRENT_JOB_ID = None
CURRENT_PROCESS = None
CURRENT_OUTPUT_PATH = None
CURRENT_SCENE_INDEX = -1
# SCENES_DATA will store: {'prompt': str, 'video_path': str, 'audio_path': str, 'first_frame': str, 'last_frame': str}
SCENES_DATA = [None] * 20  # Increased to 20 scenes support

# --- Logic Functions ---

def extract_frame(video_path, output_image_path, frame_idx=0):
    """Extracts a specific frame by index using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle negative indices (like -1 for last)
    if frame_idx < 0:
        frame_idx = frame_count + frame_idx
    
    # Bounds check
    if frame_idx >= frame_count or frame_idx < 0:
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_image_path, frame)
        cap.release()
        return True
    cap.release()
    return False

def extract_first_frame(video_path, output_image_path):
    return extract_frame(video_path, output_image_path, 0)

def extract_last_frame(video_path, output_image_path):
    return extract_frame(video_path, output_image_path, -1)

def slice_audio(audio_path, prompt, fps, num_frames):
    global SCENES_DATA
    if not audio_path:
        return "Please upload an audio file.", []
    
    try:
        os.makedirs(AUDIO_CLIPS_DIR, exist_ok=True)
        
        # Load audio info
        info = torchaudio.info(audio_path)
        sample_rate = info.sample_rate
        total_frames = info.num_frames
        duration_sec = total_frames / sample_rate
        
        # Calculate video scene duration
        scene_duration_sec = num_frames / fps
        
        num_scenes = math.ceil(duration_sec / scene_duration_sec)
        num_scenes = min(num_scenes, 20) # Limit to 20 scenes
        
        waveform, sr = torchaudio.load(audio_path)
        
        new_scenes_data = [None] * 20
        ui_updates = []
        
        samples_per_scene = int(scene_duration_sec * sr)
        
        for i in range(num_scenes):
            start_sample = i * samples_per_scene
            end_sample = min((i + 1) * samples_per_scene, total_frames)
            
            chunk_waveform = waveform[:, start_sample:end_sample]
            
            # Save chunk
            chunk_filename = f"scene_{i+1}_audio.wav"
            chunk_path = os.path.join(AUDIO_CLIPS_DIR, chunk_filename)
            torchaudio.save(chunk_path, chunk_waveform, sr)
            
            new_scenes_data[i] = {
                'prompt': prompt, # Copy master prompt
                'video_path': None,
                'audio_path': os.path.abspath(chunk_path),
                'first_frame': None,
                'last_frame': None
            }
            
        SCENES_DATA = new_scenes_data
        
        # Prepare UI updates
        for i in range(20):
            if i < num_scenes:
                # Row visible, Textbox updated
                ui_updates.append(gr.update(visible=True)) # Row
                ui_updates.append(gr.update(value=prompt, visible=True)) # Textbox
                ui_updates.append(gr.update(value=new_scenes_data[i]['audio_path'])) # Audio path display
                ui_updates.append(gr.update(value=None)) # Start Image
                ui_updates.append(gr.update(value=None)) # Last Image Override
            else:
                ui_updates.append(gr.update(visible=False)) # Row
                ui_updates.append(gr.update(visible=False)) # Textbox
                ui_updates.append(gr.update(value=None)) # Audio
                ui_updates.append(gr.update(value=None)) # Start Image
                ui_updates.append(gr.update(value=None)) # Last Image Override
        
        return tuple([f"Sliced into {num_scenes} scenes."] + ui_updates)
        
    except Exception as e:
        print(f"DEBUG Error in slice_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return tuple([f"Error: {str(e)}"] + [gr.update(visible=False)] * 20 * 5) # Update this count if UI structure changes

def process_chain_generation(scenes_list, audios_list, start_images_list, last_images_list, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, start_index=None, mode="forward"):
    """
    mode: "forward" (chain from start_index up to end) or "single" (only start_index)
    """
    global CURRENT_LOG, LATEST_VIDEO_PATH, CURRENT_PROCESS, CURRENT_OUTPUT_PATH, IS_PROCESSING, STOP_GENERATION, SCENES_DATA

    IS_PROCESSING = True
    STOP_GENERATION = False
    
    # scenes_list is a list of prompts (or None for empty slots)
    valid_indices = [i for i, p in enumerate(scenes_list) if p]
    if not valid_indices:
        IS_PROCESSING = False
        return

    if mode == "single":
        indices_to_process = [start_index]
    else:
        # Forward chain from start_index (or the first valid index) up to the end
        current_start = start_index if start_index is not None else valid_indices[0]
        indices_to_process = [i for i in range(current_start, len(scenes_list)) if i in valid_indices]

    for i in indices_to_process:
        global CURRENT_SCENE_INDEX
        CURRENT_SCENE_INDEX = i
        if STOP_GENERATION:
            CURRENT_LOG += "\n--- STOPPED BY USER ---\n"
            break
            
        prompt = scenes_list[i]
        scene_data = SCENES_DATA[i]
        
        if not scene_data:
             CURRENT_LOG += f"\nSkipping scene {i+1} : No Data\n"
             continue
             
        audio_path = audios_list[i] or scene_data.get('audio_path')
        scene_id = i + 1
        CURRENT_LOG += f"\n\n--- GENERATING SCENE {scene_id} ---\n"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"scene_{scene_id}_{timestamp}.mp4"
        output_path = os.path.abspath(output_filename)
        CURRENT_OUTPUT_PATH = output_path
        
        # --- Context & Continuity Setup ---
        actual_prompt = prompt
        actual_num_frames = int(num_frames)
        conditioning_frames = []
        
        current_seed = seed
        if random_seed:
            current_seed = int(os.urandom(4).hex(), 16) % (2 ** 32)

        # Standard Continuity (Music Video likely doesn't need context compression as much as continuity? Let's keep Standard for now)
        def get_valid_path(img_data):
            if not img_data: return None
            if isinstance(img_data, str) and img_data.strip(): return img_data
            if isinstance(img_data, dict) and 'path' in img_data and img_data['path']: return img_data['path']
            return None

        custom_start = get_valid_path(start_images_list[i])
        
        if custom_start:
            conditioning_frames = [(custom_start, 0, 1.0)]
            CURRENT_LOG += f"Using custom Start Image for Scene {scene_id}\n"
        elif mode == "forward":
            if i > 0:
                custom_last = get_valid_path(last_images_list[i-1])
                if custom_last:
                    conditioning_frames = [(custom_last, 0, 1.0)]
                    CURRENT_LOG += f"Connecting Scene {scene_id} to Scene {i} (using custom Last Image Override)\n"
                elif SCENES_DATA[i-1] and SCENES_DATA[i-1]['video_path']:
                    prev_video = SCENES_DATA[i-1]['video_path']
                    
                    cap = cv2.VideoCapture(prev_video)
                    prev_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    last_lat = (prev_frame_cnt - 1) // 8
                    
                    f_prev_l1 = f"scene_{scene_id}_f_prev_last1.jpg"
                    if extract_frame(prev_video, f_prev_l1, last_lat*8):
                         conditioning_frames = [(f_prev_l1, 0, 1.0)]
                         CURRENT_LOG += f"Connecting Scene {scene_id} to Scene {i} (last frames as latents)\n"

        # Build Command for music_to_video.py
        cmd = [
            sys.executable, "-m", "ltx_pipelines.music_to_video",
            "--distilled-checkpoint-path", checkpoint,
            "--gemma-root", gemma,
            "--spatial-upsampler-path", upsampler,
            "--prompt", actual_prompt,
            "--output-path", output_path,
            "--width", str(width),
            "--height", str(height),
            "--num-frames", str(int(actual_num_frames)),
            "--frame-rate", str(fps),
            "--num-inference-steps", str(int(steps)),
            "--seed", str(int(current_seed)),
            "--quantization", "fp8-cast"
        ]
        
        if audio_path:
            cmd.extend(["--audio-input-path", audio_path])
            
        for frame_path, latent_idx, guidance in conditioning_frames:
            cmd.extend(["--image", frame_path, str(latent_idx), str(guidance)])

        try:
            CURRENT_PROCESS = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            for line in CURRENT_PROCESS.stdout:
                if STOP_GENERATION:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(CURRENT_PROCESS.pid)], capture_output=True)
                    break
                CURRENT_LOG += line
            CURRENT_PROCESS.wait()
            
            if CURRENT_PROCESS.returncode == 0 and os.path.exists(output_path):
                CURRENT_LOG += f"Scene {scene_id} Complete.\n"
                LATEST_VIDEO_PATH = output_path
                
                # Update SCENES_DATA
                first_f = f"scene_{scene_id}_first.jpg"
                last_f = f"scene_{scene_id}_last.jpg"
                extract_first_frame(output_path, first_f)
                extract_last_frame(output_path, last_f)
                
                SCENES_DATA[i]['video_path'] = output_path
                SCENES_DATA[i]['first_frame'] = first_f
                SCENES_DATA[i]['last_frame'] = last_f
            else:
                CURRENT_LOG += f"Scene {scene_id} Failed or Canceled.\n"
                break
        except Exception as e:
            CURRENT_LOG += f"Exception: {str(e)}\n"
            break
            
    CURRENT_LOG += "\n--- GENERATION CYCLE FINISHED ---\n"
    IS_PROCESSING = False

def start_generation_thread(prompts, audios, start_images, last_images, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, start_index=None, mode="forward"):
    # Clear subsequent scenes in data if starting a chain or single regeneration logic?
    # For music video, if we regenerate, we keep the audio_path!
    # So we should only clear video_path.
    
    if mode == "forward":
        begin_idx = start_index if start_index is not None else 0
        for i in range(begin_idx, 20):
            if SCENES_DATA[i]:
                SCENES_DATA[i]['video_path'] = None
                SCENES_DATA[i]['first_frame'] = None
                SCENES_DATA[i]['last_frame'] = None

    threading.Thread(target=process_chain_generation, args=(prompts, audios, start_images, last_images, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, start_index, mode), daemon=True).start()
    return "Generation started..."

def stop_generation():
    global STOP_GENERATION
    STOP_GENERATION = True
    return "Stopping..."

def update_ui():
    global LATEST_VIDEO_PATH, CURRENT_LOG, SCENES_DATA, CURRENT_OUTPUT_PATH, CURRENT_SCENE_INDEX
    status = "Processing..." if IS_PROCESSING else "Idle"
    
    # Prepare updates for all scene boxes
    updates = []
    for i in range(20):
        data = SCENES_DATA[i]
        
        display_video = data.get('video_path') if data else None
        display_preview = data.get('last_frame') if data else None
        
        # Intermediate preview logic
        if IS_PROCESSING and CURRENT_SCENE_INDEX == i and CURRENT_OUTPUT_PATH:
            preview_file = CURRENT_OUTPUT_PATH.replace('.mp4', '_.mp4')
            if os.path.exists(preview_file):
                display_video = preview_file
        
        if data or (IS_PROCESSING and CURRENT_SCENE_INDEX == i):
            v_val = display_video
            p_val = display_preview
            updates.append(gr.update(value=v_val, visible=True))
            updates.append(gr.update(value=p_val, visible=True))
        else:
            updates.append(gr.update(value=None)) # Video
            updates.append(gr.update(value=None)) # Image
            
    return tuple([LATEST_VIDEO_PATH, CURRENT_LOG, status] + updates)

def cancel_job():
    global CURRENT_PROCESS, CURRENT_LOG
    if CURRENT_PROCESS:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(CURRENT_PROCESS.pid)], capture_output=True)
            CURRENT_LOG += "\n--- CANCELED ---\n"
            return "Canceled."
        except:
            return "Error canceling."
    return "No active process."

# --- UI Layout ---

theme = gr.themes.Soft(primary_hue="purple").set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
)

with gr.Blocks(title="LTX-2 Music Video Maker", theme=theme) as demo:
    gr.Markdown("# 🎵 LTX-2 Music Video Maker")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_file = gr.Audio(label="Upload Music File", type="filepath")
            master_prompt = gr.Textbox(label="Visual Style / Prompt", placeholder="Cyberpunk city, neon lights, rain...", lines=2)
            
            with gr.Accordion("LTX-2 Settings", open=True):
                checkpoint = gr.Textbox(label="Checkpoint", value=DEFAULT_CHECKPOINT)
                gemma = gr.Textbox(label="Gemma Root", value=DEFAULT_GEMMA)
                upsampler = gr.Textbox(label="Upsampler", value=DEFAULT_UPSAMPLER)
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=8)
                    fps = gr.Number(label="FPS", value=24)
                with gr.Row():
                    width = gr.Number(label="Width", value=1280)
                    height = gr.Number(label="Height", value=704)
                num_frames = gr.Slider(label="Frames per Scene", minimum=9, maximum=257, step=8, value=225)
                
                slice_btn = gr.Button("🔪 Slice Music & Prepare Scenes", variant="primary")
                
                with gr.Row():
                    seed = gr.Number(label="Seed", value=10, precision=0)
                    random_seed = gr.Checkbox(label="Random Seed", value=True)
                with gr.Row():
                    # context compression not prioritized for now, keeping args for compatibility
                    use_context_compression = gr.Checkbox(label="Use Context Compression", value=False, visible=False) 
                    latent_reuse_count = gr.Slider(label="Latent Reuse", minimum=1, maximum=8, step=1, value=2, visible=False)
                    context_depth = gr.Slider(label="Context Depth", minimum=1, maximum=5, step=1, value=2, visible=False)

        with gr.Column(scale=3):
            gr.Markdown("### 🎞️ Video Scenes")
            scene_rows = []
            scene_prompts = []
            scene_audios = [] 
            scene_start_images = []
            scene_last_images = []
            scene_videos = []
            scene_previews = []
            scene_reg_chain_btns = []
            scene_reg_single_btns = []
            
            for i in range(1, 21):
                with gr.Row(visible=False) as row: # Hidden until decomposed
                    scene_rows.append(row)
                    with gr.Column(scale=3):
                        prompt_box = gr.Textbox(label=f"Scene {i} Prompt", lines=2)
                        scene_prompts.append(prompt_box)
                        audio_comp = gr.Audio(label=f"Scene {i} Audio", type="filepath", interactive=True)
                        scene_audios.append(audio_comp)
                        
                        with gr.Row():
                            start_img = gr.Image(label="Start Image (optional)", type="filepath")
                            last_img = gr.Image(label="Last Image Override (optional)", type="filepath")
                            scene_start_images.append(start_img)
                            scene_last_images.append(last_img)
                        
                        with gr.Row():
                            chain_btn = gr.Button(f"🔗 Chain From {i}", size="sm")
                            single_btn = gr.Button(f"🎯 Only {i}", size="sm")
                            scene_reg_chain_btns.append(chain_btn)
                            scene_reg_single_btns.append(single_btn)
                    
                    video_comp = gr.Video(label="Clip", scale=2)
                    preview_comp = gr.Image(label="Last Frame", scale=1) 
                    
                    scene_videos.append(video_comp)
                    scene_previews.append(preview_comp)
            
            with gr.Row():
                generate_btn = gr.Button("🚀 Start Full Forward Chain", variant="primary", size="lg")
                stop_btn = gr.Button("🛑 Stop", variant="secondary", size="lg")
                cancel_btn = gr.Button("🗑️ Kill Process", variant="stop")
            
            latest_video = gr.Video(label="Latest Generated Scene (Global View)")
            status_box = gr.Textbox(label="Status", interactive=False)
    
    with gr.Accordion("Worker Log", open=False):
        log_box = gr.Textbox(label=None, lines=10, interactive=False)

    # --- Events ---
    slice_btn.click(
        fn=slice_audio,
        inputs=[audio_file, master_prompt, fps, num_frames],
        outputs=[status_box] + [comp for tuple_5 in zip(scene_rows, scene_prompts, scene_audios, scene_start_images, scene_last_images) for comp in tuple_5]
    )
    
    def collect_prompts_and_start(*args):
        # Args structure: [prompts_list..., audios..., start_img..., last_img..., checkpoint, ..., button_args]
        # We need to slice args.
        num_scenes = 20
        prompts = args[0:20]
        audios = args[20:40]
        start_images = args[40:60]
        last_images = args[60:80]
        rest = args[80:]
        return start_generation_thread(prompts, audios, start_images, last_images, *rest)

    all_inputs = scene_prompts + scene_audios + scene_start_images + scene_last_images + [checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth]
    
    generate_btn.click(
        fn=collect_prompts_and_start,
        inputs=all_inputs,
        outputs=[status_box]
    )

    stop_btn.click(fn=stop_generation, outputs=[status_box])
    cancel_btn.click(fn=cancel_job, outputs=[status_box])

    # Per-scene buttons
    for i in range(20):
        def make_chain_fn(index):
            def chain_fn(*args):
                num_scenes = 20
                prompts = args[0:20]
                audios = args[20:40]
                start_images = args[40:60]
                last_images = args[60:80]
                rest = args[80:]
                return start_generation_thread(prompts, audios, start_images, last_images, *rest, start_index=index, mode="forward")
            return chain_fn
            
        def make_single_fn(index):
            def single_fn(*args):
                num_scenes = 20
                prompts = args[0:20]
                audios = args[20:40]
                start_images = args[40:60]
                last_images = args[60:80]
                rest = args[80:]
                return start_generation_thread(prompts, audios, start_images, last_images, *rest, start_index=index, mode="single")
            return single_fn

        scene_reg_chain_btns[i].click(
            fn=make_chain_fn(i),
            inputs=all_inputs,
            outputs=[status_box]
        )
        
        scene_reg_single_btns[i].click(
            fn=make_single_fn(i),
            inputs=all_inputs,
            outputs=[status_box]
        )
    
    timer = gr.Timer(2)
    timer.tick(
        fn=update_ui, 
        outputs=[latest_video, log_box, status_box] + [comp for zip_list in zip(scene_videos, scene_previews) for comp in zip_list]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
