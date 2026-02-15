"""
CLI Entry Point — Command-Line Interface for the ALPR System
================================================================
Main entry point for running the application from terminal.
Supports 4 commands: train, infer, evaluate, api.

Usage:
    python -m src train --config configs/train.yaml
    python -m src infer --config configs/infer.yaml --video video.mp4
    python -m src evaluate --config configs/train.yaml
    python -m src api --config configs/api.yaml
"""

import argparse
import sys
import os


def parse_args():
    """Parse command-line arguments with subcommands.
    
    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='ALPR System — Automatic License Plate Recognition',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # train
    train_parser = subparsers.add_parser('train', help='Train plate detector')
    train_parser.add_argument('--config', type=str, required=True,
                              help='Path to YAML config file')
    train_parser.add_argument('--device', type=str, default=None,
                              help='Device (cpu/cuda)')
    train_parser.add_argument('--epochs', type=int, default=None,
                              help='Number of epochs (overrides config)')
    train_parser.add_argument('--lr', type=float, default=None,
                              help='Learning rate (overrides config)')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Path to checkpoint to resume from')

    # infer
    infer_parser = subparsers.add_parser('infer', help='Run inference on video')
    infer_parser.add_argument('--config', type=str, required=True,
                              help='Path to YAML config file')
    infer_parser.add_argument('--video', type=str, required=True,
                              help='Path to input video')
    infer_parser.add_argument('--output', type=str, default='output.mp4',
                              help='Path for output video')
    infer_parser.add_argument('--device', type=str, default=None,
                              help='Device (cpu/cuda)')
    infer_parser.add_argument('--show', action='store_true',
                              help='Display video in real-time')

    # evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--config', type=str, required=True,
                             help='Path to config file')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')
    eval_parser.add_argument('--device', type=str, default=None,
                             help='Device (cpu/cuda)')

    # api
    api_parser = subparsers.add_parser('api', help='Run REST API server')
    api_parser.add_argument('--config', type=str, default=None,
                            help='Path to config file')
    api_parser.add_argument('--host', type=str, default='0.0.0.0',
                            help='Host address (default: 0.0.0.0)')
    api_parser.add_argument('--port', type=int, default=8000,
                            help='Port number (default: 8000)')

    return parser.parse_args()


def cmd_train(args):
    """Run the training pipeline."""
    import torch
    from .utils.config import ConfigManager
    from .data.plate_dataset import COCOPlateDataset
    from .data.transforms import TrainTransform, ValTransform
    from .data.data_utils import collate_fn
    from .models.plate_detector import PlateDetector
    from .training.trainer import Trainer

    # Load config and apply CLI overrides
    config_manager = ConfigManager()
    config = config_manager.load(args.config)

    if args.device:
        config['device'] = args.device
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.lr:
        config.setdefault('training', {})['lr'] = args.lr
    if args.resume:
        config.setdefault('training', {})['resume_from'] = args.resume

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets
    data_cfg = config.get('data', {})
    train_dataset = COCOPlateDataset(
        root_dir=data_cfg.get('root_dir', 'data'),
        split='train',
        transforms=TrainTransform(),
    )
    val_dataset = COCOPlateDataset(
        root_dir=data_cfg.get('root_dir', 'data'),
        split='valid',
        transforms=ValTransform(),
    )

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_cfg.get('batch_size', 4),
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 2),
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 2),
        collate_fn=collate_fn,
    )

    # Model
    model_cfg = config.get('model', {})
    detector = PlateDetector(
        device=device,
        num_classes=model_cfg.get('num_classes', 2),
    )
    model = detector.get_model()

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    results = trainer.train()
    print(f"\nTraining results: {results}")


def cmd_infer(args):
    """Run inference on a video file."""
    from .utils.config import ConfigManager
    from .inference.pipeline import ALPRPipeline
    from .inference.video_processor import VideoProcessor

    config_manager = ConfigManager()
    config = config_manager.load(args.config)
    if args.device:
        config['device'] = args.device

    pipeline = ALPRPipeline(config)
    processor = VideoProcessor(pipeline)

    results = processor.process_video(
        input_path=args.video,
        output_path=args.output,
        show=args.show,
    )
    print(f"\nInference results: {results}")


def cmd_evaluate(args):
    """Evaluate model on validation set."""
    import torch
    from .utils.config import ConfigManager
    from .data.plate_dataset import COCOPlateDataset
    from .data.transforms import ValTransform
    from .data.data_utils import collate_fn
    from .models.plate_detector import PlateDetector
    from .training.evaluator import Evaluator

    config_manager = ConfigManager()
    config = config_manager.load(args.config)
    device = args.device or config.get('device', 'cpu')

    data_cfg = config.get('data', {})
    val_dataset = COCOPlateDataset(
        root_dir=data_cfg.get('root_dir', 'data'),
        split='valid',
        transforms=ValTransform(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 2),
        collate_fn=collate_fn,
    )

    model_cfg = config.get('model', {})
    detector = PlateDetector(
        device=device,
        num_classes=model_cfg.get('num_classes', 2),
    )
    detector.load_model(args.checkpoint)
    model = detector.get_model()

    evaluator = Evaluator(device=device)
    results = evaluator.evaluate(model, val_loader, model_cfg.get('num_classes', 2))
    print(f"\nEvaluation results: {results}")


def cmd_api(args):
    """Start the FastAPI server."""
    import uvicorn
    from .api.app import create_app

    app = create_app(config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    """Main entry point — dispatch to the appropriate command."""
    args = parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'infer':
        cmd_infer(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'api':
        cmd_api(args)
    else:
        print("Please choose a command: train, infer, evaluate, api")
        print("Run: python -m src --help for usage")
        sys.exit(1)


if __name__ == '__main__':
    main()
