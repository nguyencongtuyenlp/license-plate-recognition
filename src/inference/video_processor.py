"""
Video Processor — Process Video Files with the ALPR Pipeline
================================================================
Reads video frames, runs the ALPR pipeline on each, renders
annotations (bboxes, plate text, counting line), and writes output.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
import time

from .pipeline import ALPRPipeline


class VideoProcessor:
    """Processes video files through the ALPR pipeline.
    
    Flow: read frame → pipeline.process_frame() → render annotations → write output
    
    Args:
        pipeline: Configured ALPRPipeline instance.
    """

    def __init__(self, pipeline: ALPRPipeline):
        self.pipeline = pipeline

    def process_video(self, input_path: str,
                      output_path: str = "output.mp4",
                      show: bool = False,
                      max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Process an entire video file.
        
        Args:
            input_path: Path to input video.
            output_path: Path for annotated output video.
            show: Display frames in real-time with cv2.imshow().
            max_frames: Limit number of frames to process (None = all).
        
        Returns:
            Dict with processing stats: FPS, total_frames, final_counts.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        print(f"Processing video: {input_path}")
        print(f"  Resolution: {width}x{height}, FPS: {fps}, Total: {total_frames_in}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            # Run ALPR pipeline on the frame
            result = self.pipeline.process_frame(frame)

            # Render annotations
            annotated_frame = self._render_frame(frame, result)

            writer.write(annotated_frame)

            if show:
                cv2.imshow('ALPR', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                proc_fps = frame_count / elapsed
                print(f"  Frame {frame_count}/{total_frames_in} "
                      f"({proc_fps:.1f} FPS)")

        # Release resources
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        avg_fps = frame_count / max(elapsed, 1e-6)
        final_counts = self.pipeline.counter.get_counts()

        print(f"\nDone! {frame_count} frames, {elapsed:.1f}s, {avg_fps:.1f} FPS")
        print(f"Counts: {final_counts}")

        return {
            'total_frames': frame_count,
            'processing_fps': avg_fps,
            'output_path': output_path,
            'final_counts': final_counts,
        }

    def _render_frame(self, frame: np.ndarray,
                      result: Dict[str, Any]) -> np.ndarray:
        """Draw annotations on frame (bboxes, plate text, counting line).
        
        Args:
            frame: Original BGR frame.
            result: Result dict from pipeline.process_frame().
        
        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()

        # Draw counting line (red)
        pt1, pt2 = self.pipeline.counter.line_points
        cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)

        # Draw tracked vehicles
        for vehicle in result.get('tracked_vehicles', []):
            bbox = vehicle['bbox']
            track_id = vehicle['track_id']
            x1, y1, x2, y2 = map(int, bbox)

            # Bounding box (green)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Track ID label
            label = f"ID: {track_id}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Plate text (yellow, below bbox)
            plate_text = result.get('plate_texts', {}).get(track_id, "")
            if plate_text:
                cv2.putText(annotated, plate_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw counting info (white, top-left corner)
        counts = result.get('counts', {})
        count_text = (f"Total: {counts.get('total_count', 0)} | "
                      f"In: {counts.get('count_in', 0)} | "
                      f"Out: {counts.get('count_out', 0)}")
        cv2.putText(annotated, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return annotated
