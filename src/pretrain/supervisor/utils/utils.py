import numpy as np
import pydicom
import cv2
import torch


def load_dicom_image(file_path):
    """Load a DICOM image and return it as a numpy array."""
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(np.float32)
    intercept = dicom.RescaleIntercept if 'RescaleIntercept' in dicom else 0.0
    slope = dicom.RescaleSlope if 'RescaleSlope' in dicom else 1.0
    image = image * slope + intercept
    return image, dicom.WindowCenter, dicom.WindowWidth

def resize_image(image, target_size):
    """Resize the image to the target size using OpenCV."""
    if image.shape[:2] == target_size:
        return image
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image


def normalize_image(image, min_hu, max_hu):
    """Normalize the image to the range [0, 1] based on the specified HU window."""
    image = np.clip(image, min_hu, max_hu)
    image = (image - min_hu) / (max_hu - min_hu)
    return image

def denormalize_image(image, min_hu, max_hu):
    """Denormalize the image from the range [0, 1] back to the specified HU window."""
    image = image * (max_hu - min_hu) + min_hu
    image = np.clip(image, min_hu, max_hu)
    return image


# def get_adaptive_window(image, percentile_low=1, percentile_high=99):
#     """Determine adaptive windowing values based on image percentiles."""
#     vmin = np.percentile(image, percentile_low)
#     vmax = np.percentile(image, percentile_high)
#     return vmin, vmax


def visualize_images(images, rows, cols, savepath, titles=None, cmap='gray', vmin=None, vmax=None):
    """Visualize a list of images using matplotlib."""
    import matplotlib.pyplot as plt

    num_images = len(images)
    if num_images > rows * cols:
        num_images = rows * cols
    plt.figure(figsize=(cols * 4, rows * 4))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap=cmap, vmin=vmin, vmax=vmax)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.show()


import numpy as np
import torch


def get_adaptive_window(
    image,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    min_hu: float = -1024.0,
    max_hu: float = 3072.0,
    central_fraction: float = 0.9,
    min_window: float = 200.0,
):
    """
    根据图像的统计量自适应确定窗宽窗位（vmin, vmax）。

    image: np.ndarray 或 torch.Tensor，HU 域，形状 (..., H, W)
    """
    # 1. 转 numpy
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
    else:
        img = np.asarray(image)

    # 2. 只取中心区域（避免大块空气干扰）
    if img.ndim >= 2:
        h, w = img.shape[-2:]
        ch = max(1, int(h * central_fraction))
        cw = max(1, int(w * central_fraction))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        roi = img[..., y0:y0 + ch, x0:x0 + cw]
    else:
        roi = img

    roi = roi.reshape(-1)
    roi = roi[np.isfinite(roi)]
    if roi.size == 0:
        return float(min_hu), float(max_hu)

    # 3. 百分位估计
    vmin = float(np.percentile(roi, percentile_low))
    vmax = float(np.percentile(roi, percentile_high))

    # 4. clamp 到物理 HU 范围
    vmin = max(vmin, min_hu)
    vmax = min(vmax, max_hu)

    # 5. 保证最小窗宽
    if vmax - vmin < min_window:
        mid = 0.5 * (vmin + vmax)
        half = 0.5 * min_window
        vmin = mid - half
        vmax = mid + half
        vmin = max(vmin, min_hu)
        vmax = min(vmax, max_hu)
        if vmax - vmin < min_window:
            vmin, vmax = min_hu, max_hu

    return vmin, vmax