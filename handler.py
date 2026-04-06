"""
RunPod Serverless Handler for HVA + RVM pipeline.

Input:
  {
    "image_url": "https://...",     # Character crop image URL
    "audio_url": "https://...",     # Audio file URL
    "prompt": "An illustrated...",  # HVA prompt
    "fps": 25,                      # Optional, default 25
    "infer_steps": 15,              # Optional, default 15
    "image_size": 512               # Optional, default 512
  }

Output:
  {
    "rvm_frames_b64": ["base64...", ...],  # Per-frame RGBA PNGs as base64
    "frame_count": 129,
    "fps": 25,
    "width": 512,
    "height": 704
  }
"""

import runpod
import os
import sys
import subprocess
import base64
import glob
import urllib.request

VOLUME_PATH = "/runpod-volume"
HVA_WEIGHTS = os.path.join(VOLUME_PATH, "hva_weights")
RVM_WEIGHTS = os.path.join(VOLUME_PATH, "rvm", "rvm_resnet50.pth")


def download_file(url, dest):
    urllib.request.urlretrieve(url, dest)


def ensure_weights():
    """Download weights to network volume on first run. Subsequent runs use cache."""
    if not os.path.exists(os.path.join(HVA_WEIGHTS, "ckpts")):
        print("First run — downloading HVA weights to network volume...")
        os.makedirs(HVA_WEIGHTS, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="tencent/HunyuanVideo-Avatar", local_dir=HVA_WEIGHTS)
        print("HVA weights cached.")
    else:
        print("HVA weights found on volume.")

    if not os.path.exists(RVM_WEIGHTS):
        print("First run — downloading RVM weights to network volume...")
        os.makedirs(os.path.dirname(RVM_WEIGHTS), exist_ok=True)
        import torch
        model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50")
        torch.save(model.state_dict(), RVM_WEIGHTS)
        print("RVM weights cached.")
    else:
        print("RVM weights found on volume.")

def run_hva(image_path, audio_path, prompt, fps=25, infer_steps=15, image_size=512):
    """Run HunyuanVideo-Avatar inference."""
    import csv

    hva_dir = "/app/hva"
    results_dir = "/tmp/hva_results"
    os.makedirs(results_dir, exist_ok=True)

    # Convert audio to wav if needed
    wav_path = "/tmp/input_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path, wav_path
    ], capture_output=True, check=True)

    # Write CSV input
    csv_path = "/tmp/hva_input.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["videoid", "image", "audio", "prompt", "fps"])
        writer.writerow(["1", image_path, wav_path, prompt, fps])

    # Run inference
    env = os.environ.copy()
    env["PYTHONPATH"] = hva_dir
    env["MODEL_BASE"] = HVA_WEIGHTS
    env["DISABLE_SP"] = "1"

    ckpt = os.path.join(HVA_WEIGHTS, "ckpts", "hunyuan-video-t2v-720p", "transformers", "mp_rank_00_model_states_fp8.pt")

    cmd = [
        sys.executable, os.path.join(hva_dir, "hymm_sp", "sample_gpu_poor.py"),
        "--input", csv_path,
        "--ckpt", ckpt,
        "--sample-n-frames", "129",
        "--seed", "128",
        "--image-size", str(image_size),
        "--cfg-scale", "7.5",
        "--infer-steps", str(infer_steps),
        "--use-deepcache", "1",
        "--flow-shift-eval-video", "5.0",
        "--save-path", results_dir,
        "--use-fp8",
        "--infer-min",
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HVA failed: {result.stderr[-500:]}")

    # Find output video
    videos = glob.glob(os.path.join(results_dir, "*.mp4"))
    if not videos:
        raise RuntimeError("HVA produced no output video")

    return videos[0]


def run_rvm(video_path):
    """Run RobustVideoMatting to extract per-frame RGBA."""
    sys.path.insert(0, "/app/rvm")
    import torch
    from model import MattingNetwork
    from inference import convert_video

    model = MattingNetwork("resnet50").eval().cuda()
    model.load_state_dict(torch.load(RVM_WEIGHTS))

    output_dir = "/tmp/rvm_rgba"
    os.makedirs(output_dir, exist_ok=True)

    convert_video(
        model,
        input_source=video_path,
        output_type="png_sequence",
        output_composition=output_dir,
        seq_chunk=12,
    )

    return output_dir


def handler(job):
    """RunPod serverless handler."""
    ensure_weights()
    input_data = job["input"]

    image_url = input_data["image_url"]
    audio_url = input_data["audio_url"]
    prompt = input_data.get("prompt", "A person speaking with natural movement.")
    fps = input_data.get("fps", 25)
    infer_steps = input_data.get("infer_steps", 15)
    image_size = input_data.get("image_size", 512)

    # Download inputs
    image_path = "/tmp/input_image.png"
    audio_path = "/tmp/input_audio.mp3"
    download_file(image_url, image_path)
    download_file(audio_url, audio_path)

    # Run HVA
    hva_video = run_hva(image_path, audio_path, prompt, fps, infer_steps, image_size)

    # Run RVM
    rgba_dir = run_rvm(hva_video)

    # Encode frames as base64
    frames = sorted(glob.glob(os.path.join(rgba_dir, "*.png")))
    frames_b64 = []
    for frame_path in frames:
        with open(frame_path, "rb") as f:
            frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))

    # Get video dimensions
    import cv2
    sample = cv2.imread(frames[0], cv2.IMREAD_UNCHANGED)
    h, w = sample.shape[:2]

    return {
        "rvm_frames_b64": frames_b64,
        "frame_count": len(frames_b64),
        "fps": fps,
        "width": w,
        "height": h,
    }


runpod.serverless.start({"handler": handler})
