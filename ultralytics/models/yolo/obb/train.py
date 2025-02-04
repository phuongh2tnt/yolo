# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK
import argparse


class OBBTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBTrainer

        args = dict(model="yolo11n-obb.pt", data="dota8.yaml", epochs=3)
        trainer = OBBTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a OBBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OBBModel initialized with specified config and weights."""
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


if __name__ == "__main__":
    def main():
        parser = argparse.ArgumentParser(description="Train a YOLO OBB model.")
        parser.add_argument("--data", type=str, required=True, help="Path to the dataset YAML file.")
        parser.add_argument("--cfg", type=str, required=True, help="Path to the model YAML file.")
        parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
        parser.add_argument("--batch", type=int, default=16, help="Batch size.")
        parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
        args = parser.parse_args()

        # Create an OBBTrainer instance with the parsed arguments
        overrides = dict(
            model=args.cfg,
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
        )
        trainer = OBBTrainer(DEFAULT_CFG, overrides=overrides)

        # Start training
        trainer.train()

    # Call the main function
    main()
