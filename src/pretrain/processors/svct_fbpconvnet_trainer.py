import os
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.fbpconvnet import FBPConvNet
from dataset import DegradationDataset

from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from utils.model_save_load import save_model

from utils.swanlab_utils import (
    config_swanlab,
    log_images_to_swanlab,
    log_metrics_to_swanlab,
)

# ---------------- HU 范围配置（和你的仿真一致即可） ----------------
MIN_HU = -1024.0
MAX_HU = 1024.0
HU_DATA_RANGE = MAX_HU - MIN_HU
# --------------------------------------------------------

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FBPConvNet().to(self.device)
        self.model_config = {}

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        # MSE 仍在归一化空间（0~1）上计算
        self.loss_fn = torch.nn.MSELoss().to(self.device)

        self.epochs = args.epochs
        self.save_dir = args.save_dir
        self.vis_every_n_step = args.vis_every_n_step

        self.global_step = 0  # 用于 SwanLab step

    @staticmethod
    def _to_hu(x: torch.Tensor) -> torch.Tensor:
        """
        将归一化图像还原到 HU 域（完全用 torch 计算，兼容 GPU）。
        假设 normalize_image:
            norm = (img - MIN_HU) / (MAX_HU - MIN_HU)
        则反变换:
            img = norm * (MAX_HU - MIN_HU) + MIN_HU
        """
        hu = x * (MAX_HU - MIN_HU) + MIN_HU
        hu = torch.clamp(hu, MIN_HU, MAX_HU)
        return hu


    def train_step(self, batch, batch_idx):
        degradated_images, clean_images = batch  # [B, 1, H, W]
        degradated_images = degradated_images.to(self.device)
        clean_images = clean_images.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(degradated_images)

        # 损失在归一化空间计算
        loss = self.loss_fn(outputs, clean_images)
        loss.backward()
        self.optimizer.step()

        # 在 HU 域上计算 PSNR / SSIM
        with torch.no_grad():
            outputs_hu = self._to_hu(outputs)
            clean_hu = self._to_hu(clean_images)

            psnr = peak_signal_noise_ratio(
                preds=outputs_hu,
                target=clean_hu,
                data_range=HU_DATA_RANGE,
            )
            ssim = structural_similarity_index_measure(
                preds=outputs_hu,
                target=clean_hu,
                data_range=HU_DATA_RANGE,
            )

        metrics = {
            "loss": loss.item(),
            "psnr": psnr.item(),
            "ssim": ssim.item(),
        }

        # 日志输出（控制台 / log 文件）
        logger.info(
            f"[Train] step={self.global_step} "
            f"loss={metrics['loss']:.4f}, "
            f"psnr={metrics['psnr']:.4f}, "
            f"ssim={metrics['ssim']:.4f}"
        )

        if self.vis_every_n_step and (batch_idx % self.vis_every_n_step == 0):
            # 记录到 SwanLab
            log_metrics_to_swanlab(metrics, mode="train", step=self.global_step)
            # 可视化的时候，用 HU 域图像更直观
            degradated_hu = self._to_hu(degradated_images)
            clean_hu = self._to_hu(clean_images)
            outputs_hu = self._to_hu(outputs)
            log_images_to_swanlab(
                degradated_hu,
                clean_hu,
                outputs_hu,
                mode="train",
                step=self.global_step,
            )

        self.global_step += 1
        return metrics

    def validation_step(self, batch, batch_idx=0):
        degradated_images, clean_images = batch
        degradated_images = degradated_images.to(self.device)
        clean_images = clean_images.to(self.device)

        with torch.no_grad():
            outputs = self.model(degradated_images)
            loss = self.loss_fn(outputs, clean_images)

            outputs_hu = self._to_hu(outputs)
            clean_hu = self._to_hu(clean_images)

            psnr = peak_signal_noise_ratio(
                preds=outputs_hu,
                target=clean_hu,
                data_range=HU_DATA_RANGE,
            )
            ssim = structural_similarity_index_measure(
                preds=outputs_hu,
                target=clean_hu,
                data_range=HU_DATA_RANGE,
            )

        metrics = {
            "val_loss": loss.item(),
            "val_psnr": psnr.item(),
            "val_ssim": ssim.item(),
        }

        logger.info(
            f"[Val] step={self.global_step} "
            f"loss={metrics['val_loss']:.4f}, "
            f"psnr={metrics['val_psnr']:.4f}, "
            f"ssim={metrics['val_ssim']:.4f}"
        )

        log_metrics_to_swanlab(metrics, mode="val", step=self.global_step)

        if batch_idx == 0:
            degradated_hu = self._to_hu(degradated_images)
            clean_hu = self._to_hu(clean_images)
            outputs_hu = self._to_hu(outputs)
            log_images_to_swanlab(
                degradated_hu,
                clean_hu,
                outputs_hu,
                mode="val",
                step=self.global_step,
            )

        return metrics

    def train(self, train_loader, val_loader):
        best_val_loss = float("inf")
        best_val_psnr = float("-inf")
        best_val_ssim = float("-inf")

        for epoch in range(self.epochs):
            logger.info(f"========== Epoch [{epoch+1}/{self.epochs}] ==========")
            self.model.train()
            for batch_idx, batch in enumerate(train_loader):
                _ = self.train_step(batch, batch_idx)

            self.scheduler.step()

            # 验证
            self.model.eval()
            val_losses = []
            val_psnrs = []
            val_ssims = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    metrics = self.validation_step(batch, batch_idx)
                    val_losses.append(metrics["val_loss"])
                    val_psnrs.append(metrics["val_psnr"])
                    val_ssims.append(metrics["val_ssim"])

            avg_val_loss = float(np.mean(val_losses))
            avg_val_psnr = float(np.mean(val_psnrs))
            avg_val_ssim = float(np.mean(val_ssims))

            logger.info(
                f"[Epoch {epoch+1}/{self.epochs}] "
                f"Val Loss={avg_val_loss:.4f}, "
                f"PSNR={avg_val_psnr:.4f}, "
                f"SSIM={avg_val_ssim:.4f}"
            )

            # 同时把 epoch 级别的指标也打到 SwanLab
            log_metrics_to_swanlab(
                {
                    "epoch_loss": avg_val_loss,
                    "epoch_psnr": avg_val_psnr,
                    "epoch_ssim": avg_val_ssim,
                },
                mode="val_epoch",
                step=epoch,
            )

            # 保存 best 模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_model(
                    self.model,
                    self.model_config,
                    os.path.join(self.save_dir, "best_val_loss_model.pth"),
                )
                logger.info(
                    f"Best val loss model saved at epoch {epoch+1}, "
                    f"loss={best_val_loss:.4f}"
                )

            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                save_model(
                    self.model,
                    self.model_config,
                    os.path.join(self.save_dir, "best_val_psnr_model.pth"),
                )
                logger.info(
                    f"Best val PSNR model saved at epoch {epoch+1}, "
                    f"psnr={best_val_psnr:.4f}"
                )

            if avg_val_ssim > best_val_ssim:
                best_val_ssim = avg_val_ssim
                save_model(
                    self.model,
                    self.model_config,
                    os.path.join(self.save_dir, "best_val_ssim_model.pth"),
                )
                logger.info(
                    f"Best val SSIM model saved at epoch {epoch+1}, "
                    f"ssim={best_val_ssim:.4f}"
                )

        logger.info("Training complete.")


def train(args):
    # Dataset
    train_dataset = DegradationDataset(
        dataset_info_file=args.dataset_info_file,
        split="train",
        train_ratio=args.train_ratio,
        severity=args.severity,
        degradation_type="svct",
    )

    val_dataset = DegradationDataset(
        dataset_info_file=args.dataset_info_file,
        split="test",
        train_ratio=args.train_ratio,
        severity=args.severity,
        degradation_type="svct",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,      # 可以尝试从 4 提到 8 或 12
        pin_memory=True,                   # GPU 训练建议打开
        persistent_workers=True,           # 多 epoch 时减少 worker 重启开销
        prefetch_factor=4,                 # 一次预取多个 batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    trainer = Trainer(args)
    trainer.train(train_loader, val_loader)


def main():
    parser = argparse.ArgumentParser(description="SVCT Training with SwanLab")
    parser.add_argument(
        "--dataset_info_file",
        type=str,
        default="/data/hyq/codes/AgenticCT/data/deeplesion/svct/dataset_info.csv",
        help="Path to the dataset info file",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--train_ratio", type=float, default=0.75, help="Train/validation split ratio"
    )
    parser.add_argument(
        "--severity", type=str, default="low", help="Severity level of the degradation"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--vis_every_n_step",
        type=int,
        default=100,
        help="Visualize and log images every n steps during training",
    )
    parser.add_argument(
        "--save_dir_root",
        type=str,
        default="/data/hyq/codes/AgenticCT/src/pretrain/outputs/deeplesion/svct",
        help="Root directory to save model checkpoints and logs",
    )

    args = parser.parse_args()

    # 日志和保存路径
    args.save_dir = os.path.join(args.save_dir_root, args.severity)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.save_dir_root, f"{args.severity}.log"),
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting training with configuration:")
    logger.info(args)

    # 初始化 SwanLab（替代 wandb）
    config_swanlab(
        project_name="deeplesion_svct",
        run_name=args.severity,
        config=vars(args),
        logdir=os.path.join(args.save_dir_root, "swanlab_logs"),
        mode="cloud",  # 如果想本地离线，可以改成 "local"
        description=f"SVCT FBPConvnet, severity={args.severity}",
    )

    train(args=args)


if __name__ == "__main__":
    main()
