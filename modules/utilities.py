import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

import modules.globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"

# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


# ───────────────────────────────────────────────────────────────
# FFMPEG helpers
# ───────────────────────────────────────────────────────────────
def run_ffmpeg(args: List[str]) -> bool:
    commands = [
        "ffmpeg",
        "-hide_banner",
        "-hwaccel",
        "auto",
        "-loglevel",
        modules.globals.log_level,
    ]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False


def detect_fps(target_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        target_path,
    ]
    output = subprocess.check_output(command).decode().strip().split("/")
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30.0


# ───────────────────────────────────────────────────────────────
# Frame-extraction cache helpers
# ───────────────────────────────────────────────────────────────
def _frames_exist(target_path: str) -> bool:
    """
    Return True if temp/<video-name> already contains at least one
    extracted PNG frame.
    """
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.isdir(temp_directory_path) and any(
        fname.lower().endswith(".png") for fname in os.listdir(temp_directory_path)
    )


def _original_frames_exist(target_path: str) -> bool:
    """Return True if a copy of the original frames exists."""
    temp_directory_path = get_temp_directory_path(target_path)
    orig_path = os.path.join(temp_directory_path, "originals")
    return os.path.isdir(orig_path) and any(
        fname.lower().endswith(".png") for fname in os.listdir(orig_path)
    )


def needs_frame_extraction(target_path: str) -> bool:
    """
    Determine whether we need to (re)run ffmpeg to explode the video.
    """
    if modules.globals.keep_frames:
        # If keeping frames, extraction is only required when we do not have
        # a preserved copy of the originals.
        return not _original_frames_exist(target_path)
    return not _frames_exist(target_path)


# ───────────────────────────────────────────────────────────────
# Core I/O operations
# ───────────────────────────────────────────────────────────────
def extract_frames(target_path: str) -> None:
    """
    Idempotent frame extraction – skips work when frames are already cached.
    """
    temp_directory_path = get_temp_directory_path(target_path)
    orig_directory_path = os.path.join(temp_directory_path, "originals")

    if not needs_frame_extraction(target_path):
        print(
            f"[UTIL] Skipping frame extraction – cached frames found for "
            f"'{os.path.basename(target_path)}'"
        )
        # restore pristine frames if we kept them
        if modules.globals.keep_frames and os.path.isdir(orig_directory_path):
            for f in glob.glob(os.path.join(orig_directory_path, "*.png")):
                shutil.copy2(
                    f, os.path.join(temp_directory_path, os.path.basename(f))
                )
        return

    # fresh extraction
    run_ffmpeg(
        [
            "-i",
            target_path,
            "-pix_fmt",
            "rgb24",
            os.path.join(temp_directory_path, "%04d.png"),
        ]
    )

    if modules.globals.keep_frames:
        Path(orig_directory_path).mkdir(parents=True, exist_ok=True)
        for f in glob.glob(os.path.join(temp_directory_path, "*.png")):
            shutil.copy2(f, os.path.join(orig_directory_path, os.path.basename(f)))


def create_video(target_path: str, fps: float = 30.0) -> None:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(
        [
            "-r",
            str(fps),
            "-i",
            os.path.join(temp_directory_path, "%04d.png"),
            "-c:v",
            modules.globals.video_encoder,
            "-crf",
            str(modules.globals.video_quality),
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "colorspace=bt709:iall=bt601-6-625:fast=1",
            "-y",
            temp_output_path,
        ]
    )


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(
        [
            "-i",
            temp_output_path,
            "-i",
            target_path,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-y",
            output_path,
        ]
    )
    if not done:
        move_temp(target_path, output_path)


# ───────────────────────────────────────────────────────────────
# Path & temp helpers
# ───────────────────────────────────────────────────────────────
def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), "*.png")))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not modules.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


# ───────────────────────────────────────────────────────────────
# Misc. utilities
# ───────────────────────────────────────────────────────────────
def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg"))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(
            download_directory_path, os.path.basename(url)
        )
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get("Content-Length", 0))
            with tqdm(
                total=total,
                desc="Downloading",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(
                    url,
                    download_file_path,
                    reporthook=lambda count, block_size, total_size: progress.update(
                        block_size
                    ),
                )  # type: ignore[attr-defined]


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
