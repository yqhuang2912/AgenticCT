import os
import logging
import argparse
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.models import resnet18

from utils.model_save_load import save_model

from utils.swanlab_utils import (
    config_swanlab,
    log_metrics_to_swanlab,
)


# ---------------- HU 范围配置（与你的仿真一致即可） ----------------
MIN_HU = -1024.0
MAX_HU = 1024.0
HU_DATA_RANGE = MAX_HU - MIN_HU
# ---------------------------------------------------------------

LABELS = ["none", "low", "medium", "high"]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LABEL = {i: k for k, i in LABEL2ID.items()}

logger = logging.getLogger(__name__)


# ---------------------------- Utils ----------------------------

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_hw(img: np.ndarray) -> np.ndarray:
    """Ensure (H, W). Accept (H,W), (1,H,W), (H,W,1)."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[0] == 1:
            return img[0]
        if img.shape[-1] == 1:
            return img[..., 0]
    raise ValueError(f"Unsupported npy shape {img.shape}. Expect (H,W) or (1,H,W) or (H,W,1).")


def normalize_hu_to_01(
    img: np.ndarray,
    window_center: float | None = None,
    window_width: float | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    """
    Normalize CT to [0,1].
    Priority:
      1) (window_center, window_width)
      2) (clip_min, clip_max)
      3) robust percentiles [0.5, 99.5]
    """
    x = img.astype(np.float32)

    if window_center is not None and window_width is not None:
        lo = float(window_center) - float(window_width) / 2.0
        hi = float(window_center) + float(window_width) / 2.0
    elif clip_min is not None and clip_max is not None:
        lo, hi = float(clip_min), float(clip_max)
    else:
        lo = float(np.percentile(x, 0.5))
        hi = float(np.percentile(x, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(x.min()), float(x.max() + 1e-6)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    return x


def compute_class_weights(labels: list[int], num_classes: int = 4) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    w = 1.0 / (counts + 1e-6)
    w = w / w.mean()
    return torch.from_numpy(w).float()


def macro_f1_from_confmat(cm: torch.Tensor) -> float:
    """cm: (C,C) rows=gt, cols=pred"""
    C = cm.shape[0]
    f1s = []
    for c in range(C):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(sum(f1s) / C)


# ---------------------------- Dataset ----------------------------

class CTQualityDataset(Dataset):
    """
    读取 dataset_info.csv，返回：
      x: [1,H,W] 归一化到 [0,1]
      y: dict(ldct/lact/svct) -> int(0..3)
    """

    def __init__(
        self,
        dataset_info_files: list[str],
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        base_dir: str | None = None,
        img_col: str = "path",
        ldct_col: str = "ldct",
        lact_col: str = "lact",
        svct_col: str = "svct",
        window_center: float | None = None,
        window_width: float | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
        augment: bool = False,
    ):
        super().__init__()
        assert split in ["train", "test", "val"]

        dfs = [pd.read_csv(p) for p in dataset_info_files]
        df = pd.concat(dfs, ignore_index=True)

        # sanitize labels
        for c in [ldct_col, lact_col, svct_col]:
            df[c] = df[c].astype(str).str.lower().str.strip()
            bad = ~df[c].isin(LABELS)
            if bad.any():
                examples = df.loc[bad, c].head(10).tolist()
                raise ValueError(f"Column {c} has invalid labels examples={examples}. Expect {LABELS}.")

        # resolve paths
        def resolve(p: str) -> str:
            p = str(p)
            if base_dir is not None and (not os.path.isabs(p)):
                return os.path.join(base_dir, p)
            return p

        all_paths = [resolve(p) for p in df[img_col].tolist()]
        y_ldct = [LABEL2ID[v] for v in df[ldct_col].tolist()]
        y_lact = [LABEL2ID[v] for v in df[lact_col].tolist()]
        y_svct = [LABEL2ID[v] for v in df[svct_col].tolist()]

        # split
        n = len(all_paths)
        idx = np.arange(n)
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        n_train = int(round(n * train_ratio))

        if split == "train":
            sel = idx[:n_train]
        else:
            sel = idx[n_train:]

        self.paths = [all_paths[i] for i in sel]
        self.y_ldct = [y_ldct[i] for i in sel]
        self.y_lact = [y_lact[i] for i in sel]
        self.y_svct = [y_svct[i] for i in sel]

        self.window_center = window_center
        self.window_width = window_width
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _rand_augment_np(self, x01: np.ndarray) -> np.ndarray:
        if not self.augment:
            return x01
        # mild gamma/bias jitter + small gaussian noise
        if np.random.rand() < 0.5:
            gamma = np.random.uniform(0.85, 1.15)
            x01 = np.clip(x01, 0, 1) ** gamma
        if np.random.rand() < 0.5:
            bias = np.random.uniform(-0.03, 0.03)
            x01 = np.clip(x01 + bias, 0, 1)
        if np.random.rand() < 0.3:
            sigma = np.random.uniform(0.0, 0.02)
            x01 = np.clip(x01 + np.random.normal(0, sigma, size=x01.shape).astype(np.float32), 0, 1)
        return x01

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = np.load(path)
        img = _to_hw(img)
        img01 = normalize_hu_to_01(
            img,
            window_center=self.window_center,
            window_width=self.window_width,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )
        img01 = self._rand_augment_np(img01)

        x = torch.from_numpy(img01).float().unsqueeze(0)  # [1,H,W]
        y = {
            "ldct": torch.tensor(self.y_ldct[idx], dtype=torch.long),
            "lact": torch.tensor(self.y_lact[idx], dtype=torch.long),
            "svct": torch.tensor(self.y_svct[idx], dtype=torch.long),
        }
        return x, y, path


# ---------------------------- Model ----------------------------

def _fixed_gaussian_kernel(ksize: int = 5, sigma: float = 1.0, device: str = "cpu") -> torch.Tensor:
    ax = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel.view(1, 1, ksize, ksize)


def _conv2d_same(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    k = w.shape[-1]
    pad = k // 2
    return F.conv2d(x, w, padding=pad)


def make_3ch_features(x01: torch.Tensor) -> torch.Tensor:
    """
    x01: [B,1,H,W] in [0,1]
    -> [B,3,H,W] = [img, highpass, gradmag]
    """
    device = x01.device
    gk = _fixed_gaussian_kernel(ksize=5, sigma=1.0, device=device)
    blur = _conv2d_same(x01, gk)
    high = x01 - blur

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    gx = _conv2d_same(x01, sobel_x)
    gy = _conv2d_same(x01, sobel_y)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-8)

    # normalize aux channels (per-sample)
    def norm_aux(t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        flat = t.view(B, -1)
        k = max(1, flat.shape[1] // 100)  # top 1%
        topk, _ = torch.topk(flat.abs(), k=k, dim=1, largest=True, sorted=False)
        scale = topk.mean(dim=1).view(B, 1, 1, 1) + 1e-6
        out = t / scale
        out = torch.clamp(out, -1.0, 1.0)
        out = (out + 1.0) * 0.5
        return out

    high_n = norm_aux(high)
    grad_n = torch.clamp(grad / (grad.amax(dim=(2, 3), keepdim=True) + 1e-6), 0.0, 1.0)
    return torch.cat([x01, high_n, grad_n], dim=1)


class CTQEModel(nn.Module):
    """三头 4 分类：ldct/lact/svct"""

    def __init__(self, pretrained: bool = True, num_classes: int = 4):
        super().__init__()
        m = resnet18(weights="DEFAULT" if pretrained else None)
        in_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m

        self.head_ldct = nn.Linear(in_dim, num_classes)
        self.head_lact = nn.Linear(in_dim, num_classes)
        self.head_svct = nn.Linear(in_dim, num_classes)

    def forward(self, x01_1ch: torch.Tensor) -> dict[str, torch.Tensor]:
        x3 = make_3ch_features(x01_1ch)
        feat = self.backbone(x3)
        return {
            "ldct": self.head_ldct(feat),
            "lact": self.head_lact(feat),
            "svct": self.head_svct(feat),
        }


class TemperatureScaler(nn.Module):
    """logits / T"""

    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(math.log(init_T), dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T).clamp(1e-3, 100.0)
        return logits / T

    def temperature(self) -> float:
        return float(torch.exp(self.log_T).detach().cpu().item())


# ---------------------------- Trainer ----------------------------

class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CTQEModel(pretrained=args.pretrained).to(self.device)

        # 你可以把温度也写进 model_config，保存时一起落盘
        self.model_config = {
            "labels": LABELS,
            "temps": {"ldct": 1.0, "lact": 1.0, "svct": 1.0},
        }

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        self.epochs = args.epochs
        self.save_dir = args.save_dir
        self.vis_every_n_step = args.vis_every_n_step
        self.label_smoothing = args.label_smoothing

        self.use_sampler = args.use_sampler
        self.global_step = 0

        # class weights（训练时在 train() 里根据 dataset 自动设置）
        self.w_ldct = None
        self.w_lact = None
        self.w_svct = None

    @staticmethod
    def _to_hu(x: torch.Tensor) -> torch.Tensor:
        hu = x * (MAX_HU - MIN_HU) + MIN_HU
        hu = torch.clamp(hu, MIN_HU, MAX_HU)
        return hu

    def _compute_metrics(self, logits: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> dict:
        metrics = {}
        for k in ["ldct", "lact", "svct"]:
            pred = torch.argmax(logits[k], dim=1)
            acc = (pred == y[k]).float().mean()
            metrics[f"{k}_acc"] = float(acc.item())
        metrics["avg_acc"] = float((metrics["ldct_acc"] + metrics["lact_acc"] + metrics["svct_acc"]) / 3.0)
        return metrics

    def train_step(self, batch, batch_idx):
        x, y, _ = batch  # x: [B,1,H,W]
        x = x.to(self.device)
        y = {k: v.to(self.device) for k, v in y.items()}

        self.optimizer.zero_grad()
        logits = self.model(x)

        loss_ldct = F.cross_entropy(
            logits["ldct"], y["ldct"], weight=self.w_ldct, label_smoothing=self.label_smoothing
        )
        loss_lact = F.cross_entropy(
            logits["lact"], y["lact"], weight=self.w_lact, label_smoothing=self.label_smoothing
        )
        loss_svct = F.cross_entropy(
            logits["svct"], y["svct"], weight=self.w_svct, label_smoothing=self.label_smoothing
        )
        loss = (loss_ldct + loss_lact + loss_svct) / 3.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics = {"loss": float(loss.item())}
        metrics.update(self._compute_metrics(logits, y))

        logger.info(
            f"[Train] step={self.global_step} "
            f"loss={metrics['loss']:.4f}, "
            f"acc(ldct/lact/svct)={metrics['ldct_acc']:.4f}/{metrics['lact_acc']:.4f}/{metrics['svct_acc']:.4f}"
        )

        if self.vis_every_n_step and (batch_idx % self.vis_every_n_step == 0):
            if log_metrics_to_swanlab is not None:
                log_metrics_to_swanlab(metrics, mode="train", step=self.global_step)

        self.global_step += 1
        return metrics

    @torch.no_grad()
    def validation_step(self, batch, batch_idx=0):
        x, y, _ = batch
        x = x.to(self.device)
        y = {k: v.to(self.device) for k, v in y.items()}

        logits = self.model(x)

        loss_ldct = F.cross_entropy(logits["ldct"], y["ldct"], reduction="mean")
        loss_lact = F.cross_entropy(logits["lact"], y["lact"], reduction="mean")
        loss_svct = F.cross_entropy(logits["svct"], y["svct"], reduction="mean")
        loss = (loss_ldct + loss_lact + loss_svct) / 3.0

        metrics = {"val_loss": float(loss.item())}
        metrics.update({f"val_{k}": v for k, v in self._compute_metrics(logits, y).items()})

        logger.info(
            f"[Val] step={self.global_step} "
            f"loss={metrics['val_loss']:.4f}, "
            f"avg_acc={metrics['val_avg_acc']:.4f}"
        )

        if log_metrics_to_swanlab is not None:
            log_metrics_to_swanlab(metrics, mode="val", step=self.global_step)

        return metrics

    @torch.no_grad()
    def _eval_macro_f1(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        cms = {k: torch.zeros((4, 4), dtype=torch.int64) for k in ["ldct", "lact", "svct"]}

        for x, y, _ in val_loader:
            x = x.to(self.device)
            y = {k: v.to(self.device) for k, v in y.items()}
            logits = self.model(x)
            for k in ["ldct", "lact", "svct"]:
                pred = torch.argmax(logits[k], dim=1)
                gt = y[k]
                for i in range(gt.shape[0]):
                    cms[k][gt[i].item(), pred[i].item()] += 1

        out = {}
        for k in ["ldct", "lact", "svct"]:
            out[f"{k}_macro_f1"] = macro_f1_from_confmat(cms[k].to(torch.float32))
        out["avg_macro_f1"] = float((out["ldct_macro_f1"] + out["lact_macro_f1"] + out["svct_macro_f1"]) / 3.0)
        return out

    def _fit_temperature_scaling_one_head(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        logits/labels 来自验证集，最小化 NLL: CE(logits/T, labels)
        """
        scaler = TemperatureScaler(init_T=1.0).to(self.device)
        logits = logits.to(self.device)
        labels = labels.to(self.device)

        optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(scaler(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return scaler.temperature()

    @torch.no_grad()
    def _collect_val_logits(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        store = {k: {"logits": [], "labels": []} for k in ["ldct", "lact", "svct"]}

        for x, y, _ in val_loader:
            x = x.to(self.device)
            y = {k: v.to(self.device) for k, v in y.items()}
            logits = self.model(x)
            for k in store.keys():
                store[k]["logits"].append(logits[k].detach().cpu())
                store[k]["labels"].append(y[k].detach().cpu())

        for k in store.keys():
            store[k]["logits"] = torch.cat(store[k]["logits"], dim=0)
            store[k]["labels"] = torch.cat(store[k]["labels"], dim=0)
        return store

    def fit_temperature_scaling(self, val_loader: DataLoader):
        logger.info("Fitting temperature scaling on validation set...")
        store = self._collect_val_logits(val_loader)

        temps = {}
        for k in ["ldct", "lact", "svct"]:
            T = self._fit_temperature_scaling_one_head(store[k]["logits"], store[k]["labels"])
            temps[k] = float(T)
            logger.info(f"[TempScaling] {k}: T={T:.4f}")

        self.model_config["temps"] = temps

        if log_metrics_to_swanlab is not None:
            log_metrics_to_swanlab(
                {f"temp_{k}": v for k, v in temps.items()},
                mode="calibration",
                step=self.global_step,
            )

    def _save_ckpt(self, filename: str):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)

        if save_model is not None:
            save_model(self.model, self.model_config, path)
        else:
            # fallback：如果你工程里没 save_model，就用 torch.save
            torch.save(
                {"state_dict": self.model.state_dict(), "model_config": self.model_config},
                path,
            )
        logger.info(f"Model saved: {path}")

    def train(self, train_loader, val_loader):
        best_val_loss = float("inf")
        best_val_f1 = float("-inf")  # 更适合分类的 best 指标

        for epoch in range(self.epochs):
            logger.info(f"========== Epoch [{epoch+1}/{self.epochs}] ==========")
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
                _ = self.train_step(batch, batch_idx)

            self.scheduler.step()

            # 验证
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    metrics = self.validation_step(batch, batch_idx)
                    val_losses.append(metrics["val_loss"])

            avg_val_loss = float(np.mean(val_losses))
            f1_metrics = self._eval_macro_f1(val_loader)
            avg_val_f1 = float(f1_metrics["avg_macro_f1"])

            logger.info(
                f"[Epoch {epoch+1}/{self.epochs}] "
                f"Val Loss={avg_val_loss:.4f}, "
                f"Val avg_macro_f1={avg_val_f1:.4f}, "
                f"F1(ldct/lact/svct)={f1_metrics['ldct_macro_f1']:.4f}/{f1_metrics['lact_macro_f1']:.4f}/{f1_metrics['svct_macro_f1']:.4f}"
            )

            if log_metrics_to_swanlab is not None:
                log_metrics_to_swanlab(
                    {
                        "epoch_val_loss": avg_val_loss,
                        "epoch_avg_macro_f1": avg_val_f1,
                        **{f"epoch_{k}": v for k, v in f1_metrics.items()},
                    },
                    mode="val_epoch",
                    step=epoch,
                )

            # 保存 best（loss / f1 都保存一份）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_ckpt("best_val_loss_model.pth")
                logger.info(f"Best val loss model saved at epoch {epoch+1}, loss={best_val_loss:.4f}")

            if avg_val_f1 > best_val_f1:
                best_val_f1 = avg_val_f1
                self._save_ckpt("best_val_f1_model.pth")
                logger.info(f"Best val f1 model saved at epoch {epoch+1}, f1={best_val_f1:.4f}")

        # 训练结束后：用验证集拟合温度，并保存“校准后”ckpt
        self.fit_temperature_scaling(val_loader)
        self._save_ckpt("best_calibrated_model.pth")

        logger.info("Training complete.")

    @torch.no_grad()
    def predict_one(self, npy_path: str) -> dict:
        """
        推理输出 JSON：
          degradations + confidence + prob
        自动应用 temperature scaling（来自 self.model_config["temps"]）
        """
        self.model.eval()
        img = np.load(npy_path)
        img = _to_hw(img)
        img01 = normalize_hu_to_01(img)
        x = torch.from_numpy(img01).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,H,W]

        logits = self.model(x)
        temps = self.model_config.get("temps", {"ldct": 1.0, "lact": 1.0, "svct": 1.0})

        out = {"degradations": {}, "confidence": {}, "prob": {}, "temps": temps}
        for k in ["ldct", "lact", "svct"]:
            T = float(temps.get(k, 1.0))
            probs = F.softmax(logits[k] / T, dim=1).squeeze(0)
            pred_id = int(torch.argmax(probs).item())
            out["degradations"][k] = ID2LABEL[pred_id]
            out["confidence"][k] = float(torch.max(probs).item())
            out["prob"][k] = {ID2LABEL[i]: float(probs[i].item()) for i in range(4)}
        return out


# ---------------------------- train() ----------------------------

def train(args):
    train_dataset = CTQualityDataset(
        dataset_info_files=args.dataset_info_files,
        split="train",
        train_ratio=args.train_ratio,
        seed=args.seed,
        base_dir=args.base_dir,
        img_col=args.img_col,
        ldct_col=args.ldct_col,
        lact_col=args.lact_col,
        svct_col=args.svct_col,
        window_center=args.window_center,
        window_width=args.window_width,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        augment=args.augment,
    )

    val_dataset = CTQualityDataset(
        dataset_info_files=args.dataset_info_files,
        split="test",
        train_ratio=args.train_ratio,
        seed=args.seed,
        base_dir=args.base_dir,
        img_col=args.img_col,
        ldct_col=args.ldct_col,
        lact_col=args.lact_col,
        svct_col=args.svct_col,
        window_center=args.window_center,
        window_width=args.window_width,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        augment=False,
    )

    # ---- sampler（可选）----
    sampler = None
    if args.use_sampler:
        # 用 “三头平均标签” 做一个简易的 sample weight（足够实用）
        y_avg = []
        for i in range(len(train_dataset)):
            y_avg.append(int(round((train_dataset.y_ldct[i] + train_dataset.y_lact[i] + train_dataset.y_svct[i]) / 3.0)))
        w_avg = compute_class_weights(y_avg, 4)
        sample_w = [float(w_avg[c].item()) for c in y_avg]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    trainer = Trainer(args)

    # ---- 计算每个 head 的 class weights ----
    w_ldct = compute_class_weights(train_dataset.y_ldct, 4).to(trainer.device)
    w_lact = compute_class_weights(train_dataset.y_lact, 4).to(trainer.device)
    w_svct = compute_class_weights(train_dataset.y_svct, 4).to(trainer.device)
    trainer.w_ldct, trainer.w_lact, trainer.w_svct = w_ldct, w_lact, w_svct

    logger.info(f"class_weights_ldct={w_ldct.detach().cpu().numpy().round(3).tolist()}")
    logger.info(f"class_weights_lact={w_lact.detach().cpu().numpy().round(3).tolist()}")
    logger.info(f"class_weights_svct={w_svct.detach().cpu().numpy().round(3).tolist()}")

    trainer.train(train_loader, val_loader)


# ---------------------------- main() ----------------------------

def main():
    parser = argparse.ArgumentParser(description="CT Quality Evaluator Training (ldct/lact/svct) with SwanLab")

    parser.add_argument(
        "--dataset_info_files",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to dataset_info.csv (can pass multiple csv files)",
    )
    parser.add_argument("--base_dir", type=str, default=None, help="Base dir to resolve relative npy paths")

    parser.add_argument("--img_col", type=str, default="path")
    parser.add_argument("--ldct_col", type=str, default="ldct")
    parser.add_argument("--lact_col", type=str, default="lact")
    parser.add_argument("--svct_col", type=str, default="svct")

    parser.add_argument("--window_center", type=float, default=None)
    parser.add_argument("--window_width", type=float, default=None)
    parser.add_argument("--clip_min", type=float, default=None)
    parser.add_argument("--clip_max", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained backbone")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--use_sampler", action="store_true")

    parser.add_argument("--vis_every_n_step", type=int, default=100)

    parser.add_argument(
        "--save_dir_root",
        type=str,
        default="/data/hyq/codes/AgenticCT/src/pretrain/outputs/qe_tool",
        help="Root directory to save model checkpoints and logs",
    )
    parser.add_argument("--run_name", type=str, default="qe_multihead")

    args = parser.parse_args()
    seed_everything(args.seed)

    # 日志和保存路径
    args.save_dir = os.path.join(args.save_dir_root, args.run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.save_dir_root, f"{args.run_name}.log"),
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting training with configuration:")
    logger.info(vars(args))

    # SwanLab
    if config_swanlab is not None:
        config_swanlab(
            project_name="ct_qe_tool",
            run_name=args.run_name,
            config=vars(args),
            logdir=os.path.join(args.save_dir_root, "swanlab_logs"),
            mode="cloud",
            description="CT quality evaluator tool: 3-head (ldct/lact/svct), 4-class each",
        )

    train(args=args)


if __name__ == "__main__":
    main()
