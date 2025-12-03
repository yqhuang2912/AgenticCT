import numpy as np
from skimage.metrics import structural_similarity as ssim   


def calculate_psnr(img1, img2, min_val=0.0, max_val=255.0):
    """
    计算PSNR (Peak Signal-to-Noise Ratio) 支持自定义数据范围
    :param img1: 原始图像
    :param img2: 经过处理后的图像
    :param min_val: 像素值的最小值
    :param max_val: 像素值的最大值
    :return: PSNR值
    """
    # 计算均方误差 (MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return 100  # 如果两张图完全相同，PSNR值为100
    
    # 计算PSNR，使用给定的最小值和最大值来确定数据范围
    data_range = max_val - min_val
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2, min_val=0.0, max_val=255.0):
    """
    计算SSIM (Structural Similarity Index) 支持自定义数据范围
    :param img1: 原始图像
    :param img2: 经过处理后的图像
    :param min_val: 像素值的最小值
    :param max_val: 像素值的最大值
    :return: SSIM值
    """
    # 计算SSIM，使用给定的最小值和最大值来确定数据范围
    ssim_index, _ = ssim(img1, img2, data_range=max_val - min_val, full=True)
    return ssim_index