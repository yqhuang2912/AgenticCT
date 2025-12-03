import numpy as np
import pydicom
import cv2
from PIL import Image


def load_dicom_image(file_path):
    """Load a DICOM image and return it as a numpy array."""
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(np.float32)
    intercept = dicom.RescaleIntercept if 'RescaleIntercept' in dicom else 0.0
    slope = dicom.RescaleSlope if 'RescaleSlope' in dicom else 1.0
    image = image * slope + intercept
    return image, dicom.WindowCenter, dicom.WindowWidth

def load_deep_lesion_image(file_path):
    """Load a DeepLesion image and return it as a numpy array."""
    image = Image.open(file_path)
    image = np.asarray(image).astype(np.float32) - 32768.0
    return image

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
    return image


def get_adaptive_window(image, percentile_low=1, percentile_high=99):
    """Determine adaptive windowing values based on image percentiles."""
    vmin = np.percentile(image, percentile_low)
    vmax = np.percentile(image, percentile_high)
    return vmin, vmax


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