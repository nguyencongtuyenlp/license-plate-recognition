"""
Experiment Logger — TensorBoard + CSV Logging
================================================
Logs training metrics to both TensorBoard (for real-time
visualization) and CSV (for easy post-analysis in pandas/Excel).
"""

import os
import csv
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Dual-channel logger: TensorBoard + CSV.
    
    Usage:
        logger = ExperimentLogger("runs/train_v1")
        logger.log_scalar("train/loss", 0.5, step=0)
        logger.log_scalar("val/mAP", 0.8, step=0)
        logger.close()
    
    Args:
        log_dir: Directory for log files.
    """

    def __init__(self, log_dir: str = 'runs/train'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard: run `tensorboard --logdir=runs/train` to view
        self.tb_writer = SummaryWriter(log_dir=log_dir)

        # CSV file for tabular metrics
        csv_path = os.path.join(log_dir, 'metrics.csv')
        self._csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['tag', 'value', 'step'])
        self._csv_headers_written = True

        print(f"[ExperimentLogger] Log dir: {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value to both TensorBoard and CSV.
        
        Args:
            tag: Metric name (e.g., "train/loss", "val/mAP@0.5").
            value: Metric value.
            step: Step number (usually epoch or iteration).
        """
        self.tb_writer.add_scalar(tag, value, step)
        self._csv_writer.writerow([tag, value, step])
        self._csv_file.flush()

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float],
                    step: int) -> None:
        """Log multiple scalars at once (shared TensorBoard chart).
        
        Args:
            main_tag: Main tag name (e.g., "loss").
            tag_scalar_dict: Dict of {sub_tag: value}.
            step: Step number.
        """
        self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)
        
        for sub_tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{sub_tag}"
            self._csv_writer.writerow([full_tag, value, step])
        self._csv_file.flush()

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text to TensorBoard (useful for notes or hyperparams).
        
        Args:
            tag: Tag name.
            text: Text content.
            step: Step number.
        """
        self.tb_writer.add_text(tag, text, step)

    def close(self) -> None:
        """Close all writers and release resources.
        
        Must be called when training is done to flush TensorBoard events.
        """
        self.tb_writer.close()
        self._csv_file.close()
        print("[ExperimentLogger] Logger closed.")
