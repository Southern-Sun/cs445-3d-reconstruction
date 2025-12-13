"""
VGGT CLI wrapper for reconstruction, video frame extraction, and masking.

Commands:
    reconstruct  - Call the VGGT reconstruction API with images.
    convert      - Convert a video into frames on disk.
    mask         - Apply a polygon mask to all images in a directory.
"""

#### Built-ins ####
import sys
from pathlib import Path

#### Third Party Libraries
import cv2
import numpy as np



class ImageProcessor:
    """Image and video processing helpers for the CLI"""
    
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    @classmethod
    def extract_frames(
        cls,
        video_path: Path,
        output_dir: Path,
        seconds_per_frame: float = 1.0,
        start: float | None = None,
        end: float | None = None,
    ) -> list[Path]:
        """
        Extract frames from a video into an output directory.

        Args:
            video_path: Path to input video file.
            output_dir: Destination directory for frames.
            seconds_per_frame: How many seconds per extracted frame. Defaults to 1/s
            start: Optional start time in seconds.
            end: Optional end time in seconds.

        Returns:
            List of frame file paths written.
        """
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        native_fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if native_fps <= 0.0:
            native_fps = 30.0

        # How many images to extract per second of footage
        frame_interval = native_fps * seconds_per_frame

        start_frame = 0
        end_frame = None

        if start is not None and start > 0:
            start_frame = int(start * native_fps)
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if end is not None and end > 0:
            end_frame = int(end * native_fps)

        frame_idx = start_frame
        saved_idx = 0
        frames_written: list[Path] = []

        while True:
            if end_frame is not None and frame_idx > end_frame:
                break

            ret, frame = video.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_interval != 0:
                continue

            frame_bgr = frame
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out_path = output_dir / f"frame_{saved_idx:05d}.png"
            cv2.imwrite(str(out_path), frame_rgb)
            frames_written.append(out_path)
            saved_idx += 1

            frame_idx += 1

        video.release()
        return frames_written

    @classmethod
    def apply_mask(
        cls,
        directory: Path,
        points: list[tuple[int, int]],
        fill_value: int = 255,
    ) -> None:
        """
        Apply a polygon mask to all images in a directory.

        The polygon is given in pixel coordinates and will be filled with the given fill_value 
        (255 by default) on all channels. VGGT can accept 0 or 255 as a mask value.
        """
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        if not points:
            raise ValueError("At least one polygon point is required")

        polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in cls.IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Warning: could not read image {path}, skipping", file=sys.stderr)
                continue

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)

            image[mask == 1] = fill_value

            output_directory = directory / "masked"
            output_directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_directory / path.name), image)
