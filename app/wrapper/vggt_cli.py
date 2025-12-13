"""
VGGT CLI wrapper for reconstruction, video frame extraction, and masking.

Commands:
    reconstruct  - Call the VGGT reconstruction API with images.
    convert      - Convert a video into frames on disk.
    mask         - Apply a polygon mask to all images in a directory.
"""

import argparse
import sys
from pathlib import Path

from api_client import APIClient
from image_processor import ImageProcessor

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def collect_images_from_dir(directory: Path) -> list[Path]:
    """
    Collect image files from a directory (non-recursive).
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    paths: list[Path] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argparse parser with subcommands.
    """
    parser = argparse.ArgumentParser(
        description="VGGT CLI wrapper for reconstruction, video conversion, and masking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # reconstruct
    reconstruct_parser = subparsers.add_parser(
        "reconstruct",
        help="Call the VGGT reconstruction API with images.",
    )
    group = reconstruct_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-d",
        "--dir",
        type=Path,
        help="Directory containing images to reconstruct.",
    )
    group.add_argument(
        "-f",
        "--file",
        dest="files",
        type=Path,
        nargs="+",
        help="One or more image files to reconstruct.",
    )
    reconstruct_parser.add_argument(
        "--confidence",
        type=float,
        default=50.0,
        help="Percentile of low-confidence points to drop (default: 50.0).",
    )
    reconstruct_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path for the GLB file (default: ./reconstruction.glb).",
    )

    # convert
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a video into a set of image files.",
    )
    convert_parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video file.",
    )
    convert_parser.add_argument(
        "-d",
        "--dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Output directory for extracted frames.",
    )
    convert_parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target frames-per-second for extraction (default: use native fps).",
    )
    convert_parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds within the video.",
    )
    convert_parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds within the video.",
    )

    # mask
    mask_parser = subparsers.add_parser(
        "mask",
        help="Apply a polygon mask to all images in a directory.",
    )
    mask_parser.add_argument(
        "-d",
        "--dir",
        dest="directory",
        type=Path,
        required=True,
        help="Directory containing images to mask.",
    )
    mask_parser.add_argument(
        "-p",
        "--point",
        dest="points",
        action="append",
        nargs=2,
        metavar=("X", "Y"),
        type=int,
        required=True,
        help="Polygon vertex as two integers: X Y. "
        "Specify multiple times to define the polygon.",
    )
    mask_parser.add_argument(
        "--fill",
        type=int,
        default=255,
        help="Fill value for pixels inside the mask (default: 0).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Entry point for the CLI.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "reconstruct":
        try:
            client = APIClient()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.dir:
            images = collect_images_from_dir(args.dir)
        else:
            images = list(args.files or [])

        if not images:
            print("No images found for reconstruction", file=sys.stderr)
            sys.exit(1)

        try:
            output_path = client.reconstruct(
                image_paths=images,
                confidence=args.confidence,
                output_path=args.output,
            )
        except Exception as exc:
            print(f"Reconstruction failed: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"Reconstruction saved to: {output_path}")

    elif args.command == "convert":
        processor = ImageProcessor()
        try:
            frames = processor.extract_frames(
                video_path=args.video,
                output_dir=args.output_dir,
                seconds_per_frame=args.fps,
                start=args.start,
                end=args.end,
            )
        except Exception as exc:
            print(f"Video conversion failed: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"Extracted {len(frames)} frames to: {args.output_dir}")

    elif args.command == "mask":
        processor = ImageProcessor()
        points: list[tuple[int, int]] = []
        for pair in args.points:
            x, y = pair
            points.append((x, y))

        try:
            processor.apply_mask(
                directory=args.directory,
                points=points,
                fill_value=args.fill,
            )
        except Exception as exc:
            print(f"Masking failed: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"Applied mask to images in: {args.directory}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
