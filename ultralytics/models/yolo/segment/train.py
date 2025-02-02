# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationTrainer

        args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml", epochs=3)
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = SegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png
#Moi cho them
if __name__ == "__main__":
    import argparse
    from ultralytics.cfg import DEFAULT_CFG

    def main():
        parser = argparse.ArgumentParser(description="Train a YOLO segmentation model.")
        parser.add_argument("--data", type=str, required=True, help="Path to the dataset YAML file.")
        parser.add_argument("--cfg", type=str, required=True, help="Path to the model YAML file.")
        parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
        parser.add_argument("--batch", type=int, default=16, help="Batch size.")
        parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
        args = parser.parse_args()

        # Create a SegmentationTrainer instance with the parsed arguments
        overrides = dict(
            model=args.cfg,
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
        )
        trainer = SegmentationTrainer(DEFAULT_CFG, overrides=overrides)
        
        # Start training
        trainer.train()

    # Call the main function
    main()
