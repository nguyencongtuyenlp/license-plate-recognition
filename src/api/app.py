"""
FastAPI Application — REST API for the ALPR Pipeline
=======================================================
Exposes HTTP endpoints for video-based license plate recognition.
Uses factory pattern for flexible app creation (dev/prod/test).
"""

import os
import tempfile
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from ..inference.pipeline import ALPRPipeline
from ..inference.video_processor import VideoProcessor
from ..utils.config import ConfigManager


def create_app(config_path: str = None) -> FastAPI:
    """Factory function to create a configured FastAPI app.
    
    Args:
        config_path: Path to YAML config file (optional).
    
    Returns:
        FastAPI application instance.
    """
    app = FastAPI(
        title="ALPR API",
        description="Automatic License Plate Recognition API",
        version="1.0.0",
    )

    # Load config
    if config_path:
        config_manager = ConfigManager()
        config = config_manager.load(config_path)
    else:
        config = {
            'device': 'cpu',
            'vehicle_conf': 0.5,
            'plate_conf': 0.5,
            'counting_line': {'start': [0, 500], 'end': [1920, 500]},
        }

    # Initialize pipeline (stored for endpoint access)
    pipeline = ALPRPipeline(config)
    processor = VideoProcessor(pipeline)

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint for monitoring."""
        return {"status": "ok"}

    @app.post("/infer_video")
    async def infer_video(file: UploadFile = File(...)) -> JSONResponse:
        """Upload a video, run ALPR pipeline, return results.
        
        Accepts video via multipart upload, processes through the
        full pipeline, and returns counting + plate recognition results.
        
        Args:
            file: Uploaded video file.
        
        Returns:
            JSON with processing results.
        """
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            output_path = tmp_path.replace(suffix, f'_output{suffix}')
            results = processor.process_video(
                input_path=tmp_path,
                output_path=output_path,
                show=False,
            )

            return JSONResponse(content={
                "status": "success",
                "total_frames": results['total_frames'],
                "processing_fps": round(results['processing_fps'], 2),
                "counts": results['final_counts'],
            })

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)}
            )

        finally:
            # Clean up temp files
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return app
