"""
Quick Demo Script - YOLOv8 License Plate Detection
===================================================
Demonstrates inference with trained YOLOv8 model on images/video.

Usage:
    # Single image
    python demo.py --image path/to/image.jpg
    
    # Video
    python demo.py --video path/to/video.mp4 --output output.mp4
    
    # Webcam
    python demo.py --webcam
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO


def demo_image(model, image_path, save_path=None):
    """Run detection on a single image."""
    print(f"🖼️  Processing image: {image_path}")
    
    # Detect
    results = model.predict(
        source=image_path,
        conf=0.5,
        augment=True,  # TTA for better accuracy
        save=save_path is None,  # Auto-save if no output specified
    )
    
    # Display results
    for r in results:
        # Print detections
        boxes = r.boxes
        print(f"✅ Found {len(boxes)} license plates")
        for box in boxes:
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            print(f"   📍 Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), "
                  f"Confidence: {conf:.2%}")
        
        # Save if specified
        if save_path:
            im_array = r.plot()
            cv2.imwrite(save_path, im_array)
            print(f"💾 Saved result to: {save_path}")
    
    return results


def demo_video(model, video_path, output_path=None):
    """Run detection on a video."""
    print(f"🎥 Processing video: {video_path}")
    
    # Detect and track
    results = model.track(
        source=video_path,
        conf=0.5,
        save=output_path is None,
        tracker='botsort.yaml',  # Built-in tracker
    )
    
    if output_path:
        print(f"💾 Saved result to: {output_path}")
    
    return results


def demo_webcam(model):
    """Run detection on webcam."""
    print("📹 Starting webcam detection (Press 'q' to quit)")
    
    results = model.predict(
        source=0,  # Webcam
        conf=0.5,
        show=True,  # Show real-time
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 License Plate Demo')
    
    # Input source
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to input image')
    group.add_argument('--video', type=str, help='Path to input video')
    group.add_argument('--webcam', action='store_true', help='Use webcam')
    
    # Model weights
    parser.add_argument('--weights', type=str,
                        default='training_artifacts/yolov8n_baseline/best.pt',
                        help='Path to model weights')
    
    # Output
    parser.add_argument('--output', type=str, help='Path to save output')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help='Computing device')
    
    args = parser.parse_args()
    
    # Load model
    print(f"🚀 Loading YOLOv8 model: {args.weights}")
    model = YOLO(args.weights)
    model.to(args.device)
    
    # Run demo
    if args.image:
        demo_image(model, args.image, args.output)
    elif args.video:
        demo_video(model, args.video, args.output)
    elif args.webcam:
        demo_webcam(model)
    
    print("\n✅ Demo complete!")


if __name__ == '__main__':
    main()
