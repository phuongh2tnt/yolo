# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn
import argparse

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

if __name__ == "__main__":
    def main():
        parser = argparse.ArgumentParser(description="Train a YOLO detection model.")
        parser.add_argument("--data", type=str, required=True, help="Path to the dataset YAML file.")
        parser.add_argument("--cfg", type=str, required=True, help="Path to the model YAML file.")
        parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
        parser.add_argument("--batch", type=int, default=16, help="Batch size.")
        parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
        args = parser.parse_args()

        overrides = dict(
            model=args.cfg,
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
        )
        trainer = DetectionTrainer(overrides=overrides)
        trainer.train()

    main()
