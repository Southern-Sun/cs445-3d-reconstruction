"""
VGGT CLI wrapper for reconstruction, video frame extraction, and masking.

Commands:
    reconstruct  - Call the VGGT reconstruction API with images.
    convert      - Convert a video into frames on disk.
    mask         - Apply a polygon mask to all images in a directory.
"""

#### Built-ins ####
import os
from pathlib import Path
from typing import Any

#### Third-Party Libraries ####
import requests
from dotenv import load_dotenv

load_dotenv()


class APIClient:
    """API client for the VGGT reconstruction endpoint."""
    
    MIMETYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }

    def __init__(
        self,
        base_url: str | None = None,
        api_token: str | None = None,
        timeout: int = 3600
    ) -> None:
        self.base_url = (base_url or os.getenv("VGGT_API_BASE_URL") or "").rstrip("/")
        self.api_token = api_token or os.getenv("VGGT_API_TOKEN")
        self.timeout = timeout

        if not self.base_url:
            raise RuntimeError("VGGT_API_BASE_URL is not set")
        if not self.api_token:
            raise RuntimeError("VGGT_API_TOKEN is not set")
        
    def ping(self) -> bool:
        """Hit the /ping status endpoint to wake the container and discover if the API is healthy"""
        url = f"{self.base_url}/ping"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        response = requests.get(url, headers=headers)
        try:
            response.raise_for_status()
            return response.json().get("status") == "healthy"
        except Exception as e:
            print(e)
            return False

    def reconstruct(
        self,
        image_paths: list[Path],
        confidence: float = 50.0,
        output_path: Path | None = None,
    ) -> Path:
        """
        Call the /reconstruct endpoint with the given image files.

        Args:
            image_paths: List of existing image file paths.
            confidence: Percentile of low-confidence points to drop.
            output_path: Optional explicit GLB output path.
                         If None, a default is created in the working directory.

        Returns:
            Path to the saved GLB file.
        """
        if not image_paths:
            raise ValueError("No image paths provided for reconstruction")

        url = f"{self.base_url}/reconstruct"
        params = {"confidence_threshhold": confidence}
        headers = {"Authorization": f"Bearer {self.api_token}"}

        files: list[tuple[str, tuple[str, Any, str]]] = []
        for path in image_paths:
            if not path.is_file() or path.suffix not in self.MIMETYPES:
                raise FileNotFoundError(f"Missing or unsupported image: {path}")
            
            files.append(
                (
                    "files",
                    (path.name, path.open("rb"), self.MIMETYPES.get(path.suffix)),
                )
            )

        # Before making the real request, ping the container to make sure it's up and ready to 
        # receive our files
        if not self.ping():
            raise RuntimeError("Containers are unhealthy!")

        response = requests.post(
            url,
            headers=headers,
            params=params,
            files=files,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"Reconstruct request failed: {response.status_code} {response.text}"
            )

        if output_path is None:
            output_path = Path.cwd() / "reconstruction.glb"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as fh:
            fh.write(response.content)

        return output_path
