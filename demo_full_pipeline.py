"""
Full ALPR Pipeline Demo
========================
Demonstrates complete license plate recognition system:
- Vehicle detection (YOLOv8)
- Multi-object tracking (SORT)
- Line crossing counting
- License plate detection (YOLOv8)
- OCR (EasyOCR)

Usage:
    python demo_full_pipeline.py --video input.mp4 --output output.mp4
"""

import argparse
import cv2
import torch
from pathlib import Path

# Import ALPR components
from src.inference.pipeline import ALPRPipeline
from src.utils.config import ConfigManager


def main():
    parser = argparse.ArgumentParser(description='Full ALPR Pipeline Demo')
    
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default='output_alpr.mp4',
                        help='Path to output video')
    parser.add_argument('--config', type=str, default='configs/inference.yaml',
                        help='Path to inference config')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'], help='Computing device')
    parser.add_argument('--show', action='store_true',
                        help='Display video in real-time')
    
    args = parser.parse_args()
    
    # Load config
    print("🔧 Loading configuration...")
    config_manager = ConfigManager()
    config = config_manager.load(args.config)
    config['device'] = args.device
    
    # Override plate detector to use YOLOv8
    config['plate_detector']['type'] = 'yolo_plate'
    config['plate_detector']['model_path'] = 'training_artifacts/yolov8n_baseline/best.pt'
    
    # Initialize pipeline
    print("🚀 Initializing ALPR pipeline...")
    pipeline = ALPRPipeline(config)
    
    # Open video
    print(f"🎥 Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {args.video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Video writer
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"💾 Output will be saved to: {args.output}")
    
    # Process video
    print("\n🔥 Processing frames...")
    frame_count = 0
    detected_plates = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame through pipeline
            result = pipeline.process_frame(frame)
            
            # Collect plate readings
            for detection in result.get('plate_detections', []):
                if detection.get('ocr_text'):
                    detected_plates.append({
                        'frame': frame_count,
                        'plate': detection['ocr_text'],
                        'confidence': detection.get('ocr_confidence', 0),
                        'vehicle_id': detection.get('track_id'),
                    })
            
            # Draw visualizations
            annotated_frame = result.get('annotated_frame', frame)
            
            # Write to output
            if args.output:
                out.write(annotated_frame)
            
            # Display
            if args.show:
                cv2.imshow('ALPR Demo', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Stopped by user")
                    break
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                      f"{len(detected_plates)} plates detected")
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if args.output:
            out.release()
        cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total plates detected: {len(detected_plates)}")
    print(f"Unique vehicles: {len(set(p['vehicle_id'] for p in detected_plates if p['vehicle_id']))}")
    
    if detected_plates:
        print("\n🚗 Detected License Plates:")
        unique_plates = {}
        for p in detected_plates:
            plate_text = p['plate']
            if plate_text not in unique_plates:
                unique_plates[plate_text] = p
        
        for i, (plate, info) in enumerate(unique_plates.items(), 1):
            print(f"  {i}. {plate} (confidence: {info['confidence']:.2%}, "
                  f"first seen: frame {info['frame']})")
    
    print("="*60)
    
    if args.output:
        print(f"\n✅ Video saved to: {args.output}")
    
    print("\n🎉 Demo complete!")


if __name__ == '__main__':
    main()
