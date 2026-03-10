import gradio as gr
import subprocess
import os
import datetime
import threading
import json
import sys
import google.generativeai as genai
from collections import deque
import cv2  # For frame extraction

# --- Configuration & Defaults ---
DEFAULT_CHECKPOINT = "./models/ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA = "./models/gemma3"
DEFAULT_UPSAMPLER = "./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

# --- Master Prompt ---
SYSTEM_INSTRUCTION = """
You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation script split into 5-8 short scenes (5 seconds each).
Each scene must guide a text-to-video model with specific visuals and integrated audio.

#### Crucial Generation Context
- We generate scenes in CHRONOLOGICAL ORDER (starting from the first scene and moving towards the last).
- The FIRST SCENE must be the MOST DETAILED, describing the environment, primary characters, and lighting with high precision to set the standard for the entire chain.
- Subsequent scenes should maintain this description while focusing on their specific action and ensuring continuity from the previous scene.

#### Continuity & Scene Construction
- All scenes are connected by shared end/start frames.
- Environment changes MUST OCCUR INSIDE a scene, not between scenes.
- Each scene must be a direct continuation of the previous one.
- Describe explicit CAMERA MOVEMENTS (e.g., "slow dolly in," "pan left," "handheld shake") within each scene.
- Transitions or scene changes must be described as part of the visual action within the 5-second block.

#### Guidelines
- Strictly follow all aspects of the user's raw input.
- If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.
- For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.
- NO SPEECH: Characters do not speak (this model produces video and background audio only). Describe reactions, expressions, and physical movements instead.
- Use active language: present-progressive verbs ("is walking," "is grasping").
- Maintain chronological flow within scenes: use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape integrated chronologically. Be specific (e.g., "distant thunder," "rustling leaves," "mechanical hum").
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified.
- Visual and audio only: NO non-visual/auditory senses.
- NO timestamps or cuts within a single scene.
- Each scene is a single continuous paragraph.

Examples of good prompts:
1. A warm sunny backyard. The camera starts in a tight cinematic close-up of a woman and a man in their 30s, facing each other with serious expressions. The woman, emotional and dramatic, says softly, “That’s it... Dad’s lost it. And we’ve lost Dad.”
The man exhales, slightly annoyed: “Stop being so dramatic, Jess.”
A beat. He glances aside, then mutters defensively, “He’s just having fun.”
The camera slowly pans right, revealing the grandfather in the garden wearing enormous butterfly wings, waving his arms in the air like he’s trying to take off.
He shouts, “Wheeeew!” as he flaps his wings with full commitment.
The woman covers her face, on the verge of tears. The tone is deadpan, absurd, and quietly tragic.

2. Static camera from inside the oven, looking outward through the slightly fogged glass door. Warm golden light glows around freshly baked cookies. The baker’s face fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Subtle reflections move across the glass as steam rises.
Baker (whispering dramatically): “Today… I achieve perfection.”
He leans even closer, nose nearly touching the glass.
“Golden edges. Soft center. The gods themselves will smell these cookies and weep.”
Baker: “Wait—”
(beat)
“Did I… forget the chocolate chips?”
Cut to side view — coworker pops into frame, chewing casually.
Coworker (mouth full): “Nope. You forgot the sugar.”
Quick zoom back to the baker’s horrified face, pressed against the oven door, as cookies deflate behind the glass. Steam drifts upward in slow motion.
pixar style acting and timing

3. Soft studio lighting glows across a warm-toned set. The audience murmurs faintly as the camera pans to reveal three guests seated on a couch — a middle-aged couple and the show’s host sitting across from them.
The host leans forward, voice steady but probing:
Host: “When did you first notice that your daughter, Missy, started to spiral?”
The woman’s face crumples; she takes a shaky breath and begins to cry. Her husband places a comforting hand on her shoulder, looking down before turning back toward the host.
Father (quietly, with guilt): “We… we don’t know what we did wrong.”
The studio falls silent for a moment. The camera cuts to the host, who looks gravely into the lens.
Host (to camera): “Let’s take a look at a short piece our team prepared — chronicling Missy’s downward path.”
The lights dim slightly as the camera pushes in on the mother’s tear-streaked face. The studio monitors flicker to life, beginning to play the segment as the audience holds its breath.

4. Pinocchio is sitting in an interrogation room, looking nervous, and slightly sweating. He's saying very quietly to himself "I didn't do it... I didn't do it... I'm not a murderer". Pinocchio's nose is quickly getting longer and longer. The camera is zooming in on the double sided mirror in the back of the room, The mirror is turning black as the camera approaches it, and exposes a blurry silhouette of two FBI detectives who stand in the dark lit room on the other side. One of them is saying "I'm telling you, I have a feeling something is off with this kiddo

#### Output Format (STRICT JSON)
Return a JSON list of objects. Each object MUST have:
[
  {
    "scene_index": 1,
    "prompt": "Style: ... [Full Prompt Text]"
  },
  ...
]
Do not include any other text or markdown fences.
"""

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
# SCENES_DATA will store: {'prompt': str, 'video_path': str, 'first_frame': str, 'last_frame': str}
SCENES_DATA = [None] * 10 

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

def call_gemini(api_key, user_prompt):
    global SCENES_DATA
    if not api_key:
        return "Please provide a Gemini API Key.", []
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview', system_instruction=SYSTEM_INSTRUCTION)
        response = model.generate_content(user_prompt)
        text = response.text.strip()
        
        # Clean potential markdown fences
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        scenes = json.loads(text)
        
        # Initialize SCENES_DATA
        new_scenes_data = [None] * 10
        for i, scene in enumerate(scenes):
            if i < 10:
                new_scenes_data[i] = {
                    'prompt': scene['prompt'],
                    'video_path': None,
                    'first_frame': None,
                    'last_frame': None
                }
        SCENES_DATA = new_scenes_data
        
        # Prepare UI updates
        ui_updates = []
        for i in range(10): 
            if i < len(scenes):
                # Row visible, Textbox updated
                ui_updates.append(gr.update(visible=True)) # Row
                ui_updates.append(gr.update(value=scenes[i]['prompt'], visible=True)) # Textbox
            else:
                ui_updates.append(gr.update(visible=False)) # Row
                ui_updates.append(gr.update(visible=False)) # Textbox
        
        return tuple(["Story decomposed into scenes."] + ui_updates)
    except Exception as e:
        print(f"DEBUG Error in call_gemini: {str(e)}")
        return tuple([f"Error: {str(e)}"] + [gr.update(visible=False)] * 10)

def process_chain_generation(scenes_list, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, start_index=None, mode="forward"):
    """
    mode: "forward" (chain from start_index up to end) or "backward" (chain from start_index down to 0) or "single" (only start_index)
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
        orig_start_idx = start_index
    elif mode == "backward":
        # Backward chain from start_index (or the last valid index) down to 0
        current_start = start_index if start_index is not None else valid_indices[-1]
        indices_to_process = [i for i in range(current_start, -1, -1) if i in valid_indices]
        orig_start_idx = current_start
    else:
        # Forward chain from start_index (or the first valid index) up to the end
        current_start = start_index if start_index is not None else valid_indices[0]
        indices_to_process = [i for i in range(current_start, len(scenes_list)) if i in valid_indices]
        orig_start_idx = current_start

    for i in indices_to_process:
        global CURRENT_SCENE_INDEX
        CURRENT_SCENE_INDEX = i
        if STOP_GENERATION:
            CURRENT_LOG += "\n--- STOPPED BY USER ---\n"
            break
            
        prompt = scenes_list[i]
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

        if use_context_compression and mode == "forward" and i > 0:
            combined_prompt_parts = []
            total_reused_latents = 0
            
            start_j = max(0, i - int(context_depth))
            for j in range(start_j, i):
                if SCENES_DATA[j] and SCENES_DATA[j]['video_path']:
                    n_j = max(1, int(latent_reuse_count) - (i - 1 - j))
                    prev_video = SCENES_DATA[j]['video_path']
                    
                    # Extract last n_j latents from prev_video
                    cap = cv2.VideoCapture(prev_video)
                    prev_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    last_latent_idx_j = (prev_frame_count - 1) // 8
                    
                    prompt_j = SCENES_DATA[j]['prompt']
                    latent_range_str = f"{total_reused_latents}-{total_reused_latents + n_j - 1}" if n_j > 1 else f"{total_reused_latents}"
                    combined_prompt_parts.append(f"Prev context for {latent_range_str} latent: {prompt_j}")
                    
                    for latent_offset in range(n_j):
                        target_latent_idx_in_prev = last_latent_idx_j - (n_j - 1 - latent_offset)
                        frame_idx_in_prev = target_latent_idx_in_prev * 8
                        
                        tmp_frame_path = f"scene_{scene_id}_ctx_{j}_{latent_offset}.jpg"
                        if extract_frame(prev_video, tmp_frame_path, frame_idx_in_prev):
                            guidance = 1.0 if (total_reused_latents + latent_offset) == 0 else 0.1
                            conditioning_frames.append((tmp_frame_path, total_reused_latents + latent_offset, guidance))
                    
                    total_reused_latents += n_j
            
            if combined_prompt_parts:
                actual_prompt = ", ".join(combined_prompt_parts) + ", Current scene: " + prompt
                actual_num_frames = int(num_frames) + total_reused_latents * 8
                CURRENT_LOG += f"Context Compression: Added {total_reused_latents} latents from {len(combined_prompt_parts)} prev scenes.\n"
        else:
            # Standard Continuity (non-context-compression)
            max_latent = int(num_frames) // 8
            if mode == "forward":
                # Connect to previous scene's LAST frames (112, 120) as starting points (0, 1)
                if i > 0 and SCENES_DATA[i-1] and SCENES_DATA[i-1]['video_path']:
                    prev_video = SCENES_DATA[i-1]['video_path']
                    # We need the last 2 latents. 
                    # If scene was 121 frames, last latents are usually 14 and 15 (if indexed from 0).
                    # Let's use the actual frame count to be safe.
                    cap = cv2.VideoCapture(prev_video)
                    prev_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    last_lat = (prev_frame_cnt - 1) // 8
                    
                    f_prev_l1 = f"scene_{scene_id}_f_prev_last1.jpg"
                    f_prev_l2 = f"scene_{scene_id}_f_prev_last2.jpg"
                    if extract_frame(prev_video, f_prev_l1, (last_lat-1)*8) and extract_frame(prev_video, f_prev_l2, last_lat*8):
                         conditioning_frames = [(f_prev_l1, 0, 1.0), (f_prev_l2, 1, 0.1)]
                         CURRENT_LOG += f"Connecting Scene {scene_id} to Scene {i} (last frames as latents 0,1)\n"
            elif mode == "single":
                # For SINGLE regeneration, we connect both ways if possible
                if i > 0 and SCENES_DATA[i-1] and SCENES_DATA[i-1]['video_path']:
                    prev_video = SCENES_DATA[i-1]['video_path']
                    cap = cv2.VideoCapture(prev_video)
                    prev_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    last_lat = (prev_frame_cnt - 1) // 8
                    
                    f_prev_l1 = f"scene_{i}_f_prev_last1.jpg"
                    f_prev_l2 = f"scene_{i}_f_prev_last2.jpg"
                    if extract_frame(prev_video, f_prev_l1, (last_lat-1)*8) and extract_frame(prev_video, f_prev_l2, last_lat*8):
                         conditioning_frames.append((f_prev_l1, 0, 1.0))
                         conditioning_frames.append((f_prev_l2, 1, 0.1))
                         CURRENT_LOG += f"Connecting to Prev Scene {i} end\n"
                
                # 2. End at OLD VERSION's END (bridging)
                if SCENES_DATA[i] and SCENES_DATA[i]['video_path']:
                    old_video = SCENES_DATA[i]['video_path']
                    cap = cv2.VideoCapture(old_video)
                    old_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    old_last_lat = (old_frame_cnt - 1) // 8
                    
                    f_old_l1 = f"scene_{i}_f_old_last1.jpg"
                    f_old_l2 = f"scene_{i}_f_old_last2.jpg"
                    if extract_frame(old_video, f_old_l1, (old_last_lat-1)*8) and extract_frame(old_video, f_old_l2, old_last_lat*8):
                        conditioning_frames.append((f_old_l1, max_latent-1, 1.0))
                        conditioning_frames.append((f_old_l2, max_latent, 0.1))
                        CURRENT_LOG += f"Connecting to Old Scene {i+1} end (bridging)\n"
            else:
                # Backward conditioning
                if i + 1 < len(SCENES_DATA) and SCENES_DATA[i+1] and SCENES_DATA[i+1]['first_frame']:
                    conditioning_frames = [(SCENES_DATA[i+1]['first_frame'], max_latent, 1.0)]
                    CURRENT_LOG += f"Using continuity frame from scene {i+2} (latent index {max_latent})\n"

        # Build Command
        cmd = [
            sys.executable, "-m", "ltx_pipelines.distilled",
            # "kernprof", "-l", "-v", "-m", "ltx_pipelines.distilled",
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
            
        for frame_path, latent_idx, guidance in conditioning_frames:
            cmd.extend(["--image", frame_path, str(latent_idx), str(guidance)])

        try:
            CURRENT_PROCESS = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            for line in CURRENT_PROCESS.stdout:
                if STOP_GENERATION:
                    # Try to terminate
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
                
                SCENES_DATA[i] = {
                    'prompt': prompt,
                    'video_path': output_path,
                    'first_frame': first_f,
                    'last_frame': last_f
                }
            else:
                CURRENT_LOG += f"Scene {scene_id} Failed or Canceled.\n"
                break
        except Exception as e:
            CURRENT_LOG += f"Exception: {str(e)}\n"
            break
            
    CURRENT_LOG += "\n--- GENERATION CYCLE FINISHED ---\n"
    IS_PROCESSING = False

def start_generation_thread(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, start_index=None, mode="forward"):
    global SCENES_DATA
    scenes = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
    
    # Clear subsequent scenes in data if starting a chain or single regeneration
    if mode == "forward":
        begin_idx = start_index if start_index is not None else 0
        for i in range(begin_idx, 10):
            if i < len(scenes) and scenes[i]:
                SCENES_DATA[i] = {'prompt': scenes[i], 'video_path': None, 'first_frame': None, 'last_frame': None}
            else:
                SCENES_DATA[i] = None

    threading.Thread(target=process_chain_generation, args=(scenes, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, start_index, mode), daemon=True).start()
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
    for i in range(10):
        data = SCENES_DATA[i]
        
        display_video = data.get('video_path') if data else None
        display_preview = data.get('last_frame') if data else None
        
        # If this is the active scene being generated, look for intermediate preview
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

theme = gr.themes.Soft(primary_hue="blue").set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
)

with gr.Blocks(title="LTX-2.3 Film Maker", theme=theme) as demo:
    gr.Markdown("# 🎬 LTX-2.3 CinemaMaker UI")
    
    with gr.Row():
        with gr.Column(scale=1):
            gemini_key = gr.Textbox(label="Gemini API Key", type="password")
            story_prompt = gr.Textbox(label="Whole Movie Idea", placeholder="A robot finds a lost kitten in a rainy city...", lines=4)
            decompose_btn = gr.Button("✨ Decompose into Scenes", variant="primary")
            
            with gr.Accordion("LTX-2 Settings", open=False):
                checkpoint = gr.Textbox(label="Checkpoint", value=DEFAULT_CHECKPOINT)
                gemma = gr.Textbox(label="Gemma Root", value=DEFAULT_GEMMA)
                upsampler = gr.Textbox(label="Upsampler", value=DEFAULT_UPSAMPLER)
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=12)
                    fps = gr.Number(label="FPS", value=24)
                with gr.Row():
                    width = gr.Number(label="Width", value=1536)
                    height = gr.Number(label="Height", value=1024)
                num_frames = gr.Slider(label="Frames per Scene", minimum=9, maximum=257, step=8, value=121)
                with gr.Row():
                    seed = gr.Number(label="Seed", value=10, precision=0)
                    random_seed = gr.Checkbox(label="Random Seed", value=True)
                with gr.Row():
                    use_context_compression = gr.Checkbox(label="Use Context Compression", value=False)
                    latent_reuse_count = gr.Slider(label="Latent Reuse", minimum=1, maximum=8, step=1, value=2)
                    context_depth = gr.Slider(label="Context Depth", minimum=1, maximum=5, step=1, value=2)

        with gr.Column(scale=3):
            gr.Markdown("### 🎞️ Film Scenes")
            scene_rows = []
            scene_prompts = []
            scene_videos = []
            scene_previews = []
            scene_reg_chain_btns = []
            scene_reg_single_btns = []
            
            for i in range(1, 11):
                idx = i - 1
                with gr.Row(visible=False) as row: # Hidden until decomposed
                    scene_rows.append(row)
                    with gr.Column(scale=3):
                        prompt_box = gr.Textbox(label=f"Scene {i}", lines=3)
                        scene_prompts.append(prompt_box)
                        with gr.Row():
                            chain_btn = gr.Button(f"🔗 Chain From {i}", size="sm")
                            single_btn = gr.Button(f"🎯 Only {i}", size="sm")
                            scene_reg_chain_btns.append(chain_btn)
                            scene_reg_single_btns.append(single_btn)
                    
                    video_comp = gr.Video(label="Clip", scale=2)
                    preview_comp = gr.Image(label="Ends with", scale=1) # The LAST frame of this scene
                    
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
    # Decompose into scenes updates Rows and Textboxes
    decompose_btn.click(
        fn=call_gemini,
        inputs=[gemini_key, story_prompt],
        outputs=[status_box] + [comp for zip_list in zip(scene_rows, scene_prompts) for comp in zip_list]
    )
    
    generate_btn.click(
        fn=start_generation_thread,
        inputs=scene_prompts + [checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth],
        outputs=[status_box]
    )

    stop_btn.click(fn=stop_generation, outputs=[status_box])
    cancel_btn.click(fn=cancel_job, outputs=[status_box])

    # Per-scene buttons
    for i in range(10):
        def make_chain_fn(index):
            def chain_fn(*args):
                return start_generation_thread(*args, start_index=index, mode="forward")
            return chain_fn
            
        def make_single_fn(index):
            def single_fn(*args):
                return start_generation_thread(*args, start_index=index, mode="single")
            return single_fn

        scene_reg_chain_btns[i].click(
            fn=make_chain_fn(i),
            inputs=scene_prompts + [checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth],
            outputs=[status_box]
        )
        
        scene_reg_single_btns[i].click(
            fn=make_single_fn(i),
            inputs=scene_prompts + [checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth],
            outputs=[status_box]
        )
    
    timer = gr.Timer(2)
    timer.tick(
        fn=update_ui, 
        outputs=[latest_video, log_box, status_box] + [comp for zip_list in zip(scene_videos, scene_previews) for comp in zip_list]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
