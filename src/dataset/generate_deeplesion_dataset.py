import glob
import os
import torch
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import random
import argparse
from ctlib import projection, fbp
from utils import load_dicom_image, load_deep_lesion_image, get_adaptive_window, denormalize_image, normalize_image, resize_image

N0 = 1e5  # 初始光子数
ANGLES = 2160

BASE_OPTIONS = torch.tensor([
    ANGLES,                     # views
    768,                        # dets
    256,                        # width
    256,                        # height
    0.0078125,                  # dImg (像素尺寸)
    0.0058,                     # dDet (探测器间距)
    0.0,                        # Ang0 (开始角度)
    2 * math.pi / ANGLES,       # dAng (角度间隔)
    3.5,                        # s2r (源到探测器的距离)  
    3.0,                        # d2r (探测器到图像的距离)
    0.0,                        # binshift (探测器偏移)
    0,                          # scan_type (0: 等距离风扇型)
], dtype=torch.float32, device='cuda')


DEGRADATION_SEVERITY = {
    "ldct": {
        "high": (0.05, 0.25),  # At most 75% dose reduction, 
        "medium": (0.25, 0.5), # At most 50% dose reduction
        "low": (0.5, 0.75)    # At most 25% dose reduction
    },
    "lact": {
        "high": (90, 120),  # 90 to 120 degrees
        "medium": (120, 150), # 120 to 150 degrees
        "low": (150, 180)    # 150 to 180 degrees
    },
    "svct": {
        "high": (30, 120),    # 30 to 120 views
        "medium": (120, 240), # 120 to 240 views
        "low": (240, 360)     # 240 to 360 views
    }
}

PATIENT_IDS = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310", "L333", "L506"]


def simulate_low_dose_poisson(p_fdct, alpha):
    """
    p_fdct: full-dose sinogram (line integral)
    N0:    full-dose 入射光子数
    alpha: 剂量缩放因子 (0<alpha<=1)
    """
    print("shpae of p_fdct:", p_fdct.shape)
    N0_ldct = N0 * alpha                  # 低剂量下的 N0
    # 1) full-dose 透过率 -> 低剂量 photon 期望
    lam = N0_ldct * torch.exp(-p_fdct)     # λ = N0_ldct * exp(-p)
    # 2) Poisson 噪声
    I_ldct = torch.poisson(lam)
    # 3) 反算回 line integral
    p_ldct = -torch.log(I_ldct / N0_ldct)
    return p_ldct

def simulate_low_dose_gaussian(p_fdct, alpha):
    """
    p_fdct: full-dose sinogram (line integral)
    N0:     full-dose 入射光子数
    alpha:  剂量缩放因子 (0<alpha<=1)
    """
    scale = ((1 - alpha) / alpha) ** 0.5
    # 计算噪声的标准差
    noise_std = torch.sqrt(torch.exp(p_fdct) / N0) * scale
    # 生成高斯噪声
    x = torch.randn_like(p_fdct)
    noise_add = noise_std * x
    # 合成低剂量投影
    p_ldct = p_fdct + noise_add
    return p_ldct


def simulate_sparse_view(p_fdct, every_n_views):
    """
    Simulate sparse-view CT by downsampling the sinogram.
    
    Args:
        p_fdct: Full-dose sinogram (line integral)
        every_n_views: Keep every n-th view

    Returns:
        p_svct: Sparse-view sinogram
    """
    # select every n-th view
    view_indices = torch.arange(0, ANGLES, every_n_views, device=p_fdct.device)
    p_svct = p_fdct[:, :, view_indices, :]
    return p_svct

def create_sparse_view_geometry(every_n_views) -> torch.Tensor:
    """
    Create sparse view CT geometry based on the original image shape and downsampling factor.

    Args:
        every_n_views: Downsampling factor for views
    Returns:
        torch.Tensor: Sparse view geometry
    """
    sparse_options = BASE_OPTIONS.clone()
    sparse_options[0] = ANGLES // every_n_views  # 更新 views 数量
    sparse_options[7] = 2 * math.pi / (ANGLES // every_n_views)  # 更新 dAng
    return sparse_options

def simulate_limited_angle(p_fdct, end_angle_deg):
    """
    Simulate limited-angle CT by selecting a range of angles from the sinogram. Default start angle is 0 degree.
    
    Args:
        p_fdct: Full-dose sinogram (line integral)
        end_angle_deg: Ending angle in degrees
    Returns:
        p_lact: Limited-angle sinogram
    """
    dAng = 2 * math.pi / ANGLES
    end_angle_rad = math.radians(end_angle_deg)
    end_index = int(end_angle_rad / dAng)

    p_lact = p_fdct[:, :, :end_index, :]
    return p_lact

def create_limited_angle_geometry(end_angle_deg) -> torch.Tensor:
    """
    Create limited angle CT geometry based on the original image shape and selected angle indices.

    Args:
        end_angle_deg: Ending angle in degrees
    Returns:
        torch.Tensor: Limited angle geometry
    """
    limited_options = BASE_OPTIONS.clone()
    end_angle_rad = math.radians(end_angle_deg)
    end_views = int(end_angle_rad / (2 * math.pi / ANGLES))

    limited_options[0] = end_views  # 更新 views 数量
    return limited_options

def simulate_sparse_limited_angle(p_fdct, every_n_views, end_angle_deg):
    """
    Simulate sparse-view and limited-angle CT by downsampling the sinogram and selecting a range of angles.
    
    Args:
        p_fdct: Full-dose sinogram (line integral)
        every_n_views: Keep every n-th view
        end_angle_deg: Ending angle in degrees
    Returns:
        p_slact: Sparse-view and limited-angle sinogram
    """
    dAng = 2 * math.pi / ANGLES
    end_angle_rad = math.radians(end_angle_deg)
    end_index = int(end_angle_rad / dAng)
    
    view_indices = torch.arange(0, end_index, every_n_views, device=p_fdct.device)
    p_slact = p_fdct[:, :, view_indices, :]
    return p_slact, view_indices

def create_sparse_limited_angle_geometry(view_indices) -> torch.Tensor:
    """
    Create sparse view and limited angle CT geometry based on the original image shape and downsampling factor.

    Args:
        every_n_views: Downsampling factor for views
        end_angle_deg: Ending angle in degrees
    Returns:
        torch.Tensor: Sparse view and limited angle geometry
    """
    sparse_limited_options = BASE_OPTIONS.clone()
    num_views = len(view_indices)
    if num_views == 0:
        raise ValueError("No views selected for reconstruction")
    
    # 设置正确的view数量
    sparse_limited_options[0] = num_views
    
    # 计算并设置正确的起始角度 (Ang0)
    dAng_original = BASE_OPTIONS[7]
    first_view_index = view_indices[0].item()
    actual_start_angle = first_view_index * dAng_original
    sparse_limited_options[6] = actual_start_angle  # Ang0: 起始角度
    
    # 计算并设置正确的角度间隔 (dAng)
    if num_views > 1:
        actual_dAng = (view_indices[1] - view_indices[0]).item() * dAng_original
        sparse_limited_options[7] = actual_dAng  # dAng: 角度间隔
    else:
        # 单个view的情况，保持原始角度间隔
        sparse_limited_options[7] = dAng_original
    
    return sparse_limited_options


def generate_sparse_view_samples(input_image):
    """
    Generate sparse-view CT samples from the input image.

    Args:
        input_image: Input CT image tensor of shape [1, 1, H, W]
    Returns:
        list[torch.Tensor], torch.Tensor: Sparse-view images
    """
    p_fdct = projection(input_image, BASE_OPTIONS)
    samples =  []
    labels =  []
    ldct_severities = []
    svct_severities = []
    lact_severities = []
    views = []
    degradations = []

    # single degradation
    for _ in range(9):
        for severity, (min_views, max_views) in DEGRADATION_SEVERITY["svct"].items():
            n_views = random.randint(min_views, max_views)
            every_n_views = ANGLES // n_views
            p_svct = simulate_sparse_view(p_fdct, every_n_views)
            svct_options = create_sparse_view_geometry(every_n_views)
            svct_reconstruction = fbp(p_svct, svct_options)
            samples.append(svct_reconstruction)
            labels.append(input_image)
            svct_severities.append(severity)
            lact_severities.append(None)
            ldct_severities.append(None)
            views.append(n_views)
            degradations.append(['svct'])

    # two degradations
    for _ in range(3):
        for ldct_severity, (min_alpha, max_alpha) in DEGRADATION_SEVERITY["ldct"].items():
            alpha = random.uniform(min_alpha, max_alpha)
            p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
            label = fbp(p_ldct, BASE_OPTIONS)
            for svct_severity, (min_views, max_views) in DEGRADATION_SEVERITY["svct"].items():
                n_views = random.randint(min_views, max_views)
                every_n_views = ANGLES // n_views
                p_svct = simulate_sparse_view(p_ldct, every_n_views)
                svct_options = create_sparse_view_geometry(every_n_views)
                svct_reconstruction = fbp(p_svct, svct_options)
                samples.append(svct_reconstruction)
                labels.append(label)
                ldct_severities.append(ldct_severity)
                svct_severities.append(svct_severity)
                lact_severities.append(None)
                views.append(n_views)
                degradations.append(['ldct', 'svct'])

        for lact_severity, (min_end_angle, max_end_angle) in DEGRADATION_SEVERITY["lact"].items():
            end_angle_deg = random.randint(min_end_angle, max_end_angle)
            p_lact = simulate_limited_angle(p_fdct, end_angle_deg)
            lact_options = create_limited_angle_geometry(end_angle_deg)
            label = fbp(p_lact, lact_options)
            for svct_severity, (min_views, max_views) in DEGRADATION_SEVERITY["svct"].items():
                n_views = random.randint(min_views, max_views)
                every_n_views = ANGLES // n_views
                p_lasvct, view_interval = simulate_sparse_limited_angle(p_fdct, every_n_views, end_angle_deg)
                lasvct_options = create_sparse_limited_angle_geometry(view_interval)
                lasvct_reconstruction = fbp(p_lasvct, lasvct_options)
                samples.append(lasvct_reconstruction)
                labels.append(label)
                lact_severities.append(lact_severity)
                svct_severities.append(svct_severity)
                ldct_severities.append(None)
                views.append(n_views)
                degradations.append(['lact', 'svct'])

    # three degradations
    for ldct_severity, (min_alpha, max_alpha) in DEGRADATION_SEVERITY["ldct"].items():
        alpha = random.uniform(min_alpha, max_alpha)
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        for lact_severity, (min_end_angle, max_end_angle) in DEGRADATION_SEVERITY["lact"].items():
            end_angle_deg = random.randint(min_end_angle, max_end_angle)
            p_lact = simulate_limited_angle(p_ldct, end_angle_deg)
            lact_options = create_limited_angle_geometry(end_angle_deg)
            label = fbp(p_lact, lact_options)
            for svct_severity, (min_views, max_views) in DEGRADATION_SEVERITY["svct"].items():
                n_views = random.randint(min_views, max_views)
                every_n_views = ANGLES // n_views
                p_lasvct, view_interval = simulate_sparse_limited_angle(p_ldct, every_n_views, end_angle_deg)
                lasvct_options = create_sparse_limited_angle_geometry(view_interval)
                lasvct_reconstruction = fbp(p_lasvct, lasvct_options)
                samples.append(lasvct_reconstruction)
                labels.append(label)
                ldct_severities.append(ldct_severity)
                lact_severities.append(lact_severity)
                svct_severities.append(svct_severity)
                views.append(n_views)
                degradations.append(['ldct', 'lact', 'svct'])
    return samples, labels, ldct_severities, lact_severities, svct_severities, views, degradations


def generate_limited_angle_samples(input_image):
    """
    Generate limited-angle CT samples from the input image.

    Args:
        input_image: Input CT image tensor of shape [1, 1, H, W]
    Returns:
        list[torch.Tensor], torch.Tensor: Limited-angle images
    """
    p_fdct = projection(input_image, BASE_OPTIONS)
    samples =  []
    labels = []
    ldct_severities = []
    svct_severities = []
    lact_severities = []
    end_angles = []
    degradations = []

    # single degradation
    for _ in range(9):
        for severity, (min_end_angle, max_end_angle) in DEGRADATION_SEVERITY["lact"].items():
            end_angle_deg = random.randint(min_end_angle, max_end_angle)
            p_lact = simulate_limited_angle(p_fdct, end_angle_deg)
            lact_options = create_limited_angle_geometry(end_angle_deg)
            lact_reconstruction = fbp(p_lact, lact_options)
            samples.append(lact_reconstruction)
            labels.append(input_image)
            lact_severities.append(severity)
            svct_severities.append(None)
            ldct_severities.append(None)
            end_angles.append(end_angle_deg)
            degradations.append(['lact'])
    
    # two degradations
    for _ in range(3):
        for ldct_severity, (min_alpha, max_alpha) in DEGRADATION_SEVERITY["ldct"].items():
            alpha = random.uniform(min_alpha, max_alpha)
            p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
            label = fbp(p_ldct, BASE_OPTIONS)
            for lact_severity, (min_end_angle, max_end_angle) in DEGRADATION_SEVERITY["lact"].items():
                end_angle_deg = random.randint(min_end_angle, max_end_angle)
                p_lact = simulate_limited_angle(p_ldct, end_angle_deg)
                lact_options = create_limited_angle_geometry(end_angle_deg)
                lact_reconstruction = fbp(p_lact, lact_options)
                samples.append(lact_reconstruction)
                labels.append(label)
                ldct_severities.append(ldct_severity)
                lact_severities.append(lact_severity)
                svct_severities.append(None)
                end_angles.append(end_angle_deg)
                degradations.append(['ldct', 'lact'])
        
        for lact_severity, (min_end_angle, max_end_angle) in DEGRADATION_SEVERITY["lact"].items():
            end_angle_deg = random.randint(min_end_angle, max_end_angle)
            for svct_severity, (min_views, max_views) in DEGRADATION_SEVERITY["svct"].items():
                n_views = random.randint(min_views, max_views)
                every_n_views = ANGLES // n_views

                # sample：LACT + SVCT（有限角 + 稀疏视角）
                p_lasvct, view_interval = simulate_sparse_limited_angle(
                    p_fdct, every_n_views, end_angle_deg
                )
                lasvct_options = create_sparse_limited_angle_geometry(view_interval)
                lasvct_reconstruction = fbp(p_lasvct, lasvct_options)
                samples.append(lasvct_reconstruction)

                # label：只保留 SVCT（全角度、但稀疏视角）
                p_svct = simulate_sparse_view(p_fdct, every_n_views)        # ★ 不再 limited_angle
                svct_options = create_sparse_view_geometry(every_n_views)   # ★ 用 sparse_view 的几何
                label = fbp(p_svct, svct_options)
                labels.append(label)

                lact_severities.append(lact_severity)
                svct_severities.append(svct_severity)
                ldct_severities.append(None)
                end_angles.append(end_angle_deg)
                degradations.append(['lact', 'svct'])

    # three degradations
    for ldct_severity, (min_alpha, max_alpha) in DEGRADATION_SEVERITY["ldct"].items():
        alpha = random.uniform(min_alpha, max_alpha)
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        for lact_severity, (min_end_angle, max_end_angle) in DEGRADATION_SEVERITY["lact"].items():
            end_angle_deg = random.randint(min_end_angle, max_end_angle)
            for svct_severity, (min_views, max_views) in DEGRADATION_SEVERITY["svct"].items():
                n_views = random.randint(min_views, max_views)
                every_n_views = ANGLES // n_views
                p_lasvct, view_interval = simulate_sparse_limited_angle(p_ldct, every_n_views, end_angle_deg)
                lasvct_options = create_sparse_limited_angle_geometry(view_interval)
                lasvct_reconstruction = fbp(p_lasvct, lasvct_options)
                samples.append(lasvct_reconstruction)

                p_svct = simulate_sparse_view(p_ldct, every_n_views)
                svct_options = create_sparse_view_geometry(every_n_views)
                label = fbp(p_svct, svct_options)
                labels.append(label)

                ldct_severities.append(ldct_severity)
                lact_severities.append(lact_severity)
                svct_severities.append(svct_severity)
                end_angles.append(end_angle_deg)
                degradations.append(['ldct', 'lact', 'svct'])
    return samples, labels, ldct_severities, lact_severities, svct_severities, end_angles, degradations


def generate_low_dose_samples(input_image):
    """
    Generate low-dose CT samples from the input image.

    Args:
        input_image: Input CT image tensor of shape [1, 1, H, W]
    Returns:
        list[torch.Tensor], torch.Tensor: Low-dose images
    """
    p_fdct = projection(input_image, BASE_OPTIONS)
    samples =  []
    labels = []
    ldct_severities = []
    svct_severities = []
    lact_severities = []
    alphas = []
    degradations = []

    # single degradation (3个)
    for _ in range(9):
        for severity, (min_alpha, max_alpha) in DEGRADATION_SEVERITY["ldct"].items():
            alpha = random.uniform(min_alpha, max_alpha)
            p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
            ldct_reconstruction = fbp(p_ldct, BASE_OPTIONS)
            samples.append(ldct_reconstruction)
            labels.append(input_image)
            ldct_severities.append(severity)
            svct_severities.append(None)
            lact_severities.append(None)
            alphas.append(alpha)
            degradations.append(['ldct'])

    # two degradations (9个)
    for _ in range(3):
        for ldct_severity, (min_alpha, max_alpha) in DEGRADATION_SEVERITY["ldct"].items():
            alpha = random.uniform(min_alpha, max_alpha)
            p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
            for svct_severity, (min_views, max_views) in DEGRADATION_SEVERITY["svct"].items():
                n_views = random.randint(min_views, max_views)
                every_n_views = ANGLES // n_views
                p_svct = simulate_sparse_view(p_ldct, every_n_views)
                svct_options = create_sparse_view_geometry(every_n_views)
                svct_reconstruction = fbp(p_svct, svct_options)
                samples.append(svct_reconstruction)

                p_label_svct = simulate_sparse_view(p_fdct, every_n_views)
                label_svct_options = create_sparse_view_geometry(every_n_views)
                label = fbp(p_label_svct, label_svct_options)
                labels.append(label)

                ldct_severities.append(ldct_severity)
                svct_severities.append(svct_severity)
                lact_severities.append(None)
                alphas.append(alpha)
                degradations.append(['ldct', 'svct'])

        # two degradations (9个)
        for ldct_severity, (min_alpha, max_alpha) in DEGRADATION_SEVERITY["ldct"].items():
            alpha = random.uniform(min_alpha, max_alpha)
            p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
            for lact_severity, (min_end_angle, max_end_angle) in DEGRADATION_SEVERITY["lact"].items():
                end_angle_deg = random.randint(min_end_angle, max_end_angle)
                p_lact = simulate_limited_angle(p_ldct, end_angle_deg)
                lact_options = create_limited_angle_geometry(end_angle_deg)
                lact_reconstruction = fbp(p_lact, lact_options)
                samples.append(lact_reconstruction)

                p_label_lact = simulate_limited_angle(p_fdct, end_angle_deg)
                label_lact_options = create_limited_angle_geometry(end_angle_deg)
                label = fbp(p_label_lact, label_lact_options)
                labels.append(label)

                ldct_severities.append(ldct_severity)
                lact_severities.append(lact_severity)
                svct_severities.append(None)
                alphas.append(alpha)
                degradations.append(['ldct', 'lact'])

    # three degradations (27个)
    for ldct_severity, (min_alpha, max_alpha) in DEGRADATION_SEVERITY["ldct"].items():
        alpha = random.uniform(min_alpha, max_alpha)
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        for lact_severity, (min_end_angle, max_end_angle) in DEGRADATION_SEVERITY["lact"].items():
            end_angle_deg = random.randint(min_end_angle, max_end_angle)
            for svct_severity, (min_views, max_views) in DEGRADATION_SEVERITY["svct"].items():
                n_views = random.randint(min_views, max_views)
                every_n_views = ANGLES // n_views
                p_lasvct, view_interval = simulate_sparse_limited_angle(p_ldct, every_n_views, end_angle_deg)
                lasvct_options = create_sparse_limited_angle_geometry(view_interval)
                lasvct_reconstruction = fbp(p_lasvct, lasvct_options)
                samples.append(lasvct_reconstruction)

                p_label_lasvct, lasv_view_interval = simulate_sparse_limited_angle(p_fdct, every_n_views, end_angle_deg)
                label_lasvct_options = create_sparse_limited_angle_geometry(lasv_view_interval)
                label = fbp(p_label_lasvct, label_lasvct_options)
                labels.append(label)

                ldct_severities.append(ldct_severity)
                lact_severities.append(lact_severity)
                svct_severities.append(svct_severity)
                alphas.append(alpha)
                degradations.append(['ldct', 'lact', 'svct'])
    return samples, labels, ldct_severities, lact_severities, svct_severities, alphas, degradations


def main():
    parser = argparse.ArgumentParser(description="Simulate CT imaging scenarios")
    parser.add_argument("--target_image_dir", type=str, default="/data/hyq/data/DeepLesion/Images_png/Images/Images_png", help="Path to the input CT image")
    parser.add_argument("--dataset_info_csv", type=str, default="/data/hyq/data/DeepLesion/DL_info.csv", help="Path to the DeepLesion dataset info CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the simulated sinograms and reconstructions")
    parser.add_argument("--image_size", type=int, default=256, help="Size to resize the input image (default: 256)")
    parser.add_argument("--min_hu", type=int, default=-1024, help="Minimum HU value for normalization (default: -1024)")
    parser.add_argument("--max_hu", type=int, default=3072, help="Maximum HU value for normalization (default: 3072)")
    parser.add_argument("--degradation_type", type=str, choices=["ldct", "lact", "svct"], required=True, help="Type of degradation to simulate")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_info = {
        "image_paths": [],
        "label_paths": [],
        "window_centers": [],
        "window_widths": [],
        "ldct_severities": [],
        "lact_severities": [],
        "svct_severities": [],
        "alphas": [],
        "views": [],
        "end_angles": [],
        "degradations": []
    }

    info_df = pd.read_csv(args.dataset_info_csv)

    for idx in tqdm(range(len(info_df)), desc="Processing images"):
        row = info_df.iloc[idx]
        patient_id = row['Patient_index']
        study_id = row['Study_index']
        slice_id = row['Series_ID']
        img_id = row['Key_slice_index']
        window_low = float(row['DICOM_windows'].split(',')[0])
        window_high = float(row['DICOM_windows'].split(',')[1])
        img_dir = f"{patient_id:06d}_{study_id:02d}_{slice_id:02d}"
        img_path = os.path.join(args.target_image_dir, img_dir, f"{img_id:03d}.png")
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        output_image_dir = os.path.join(args.output_dir, img_dir)
        os.makedirs(output_image_dir, exist_ok=True)

        target_image = load_deep_lesion_image(img_path)
        target_image = resize_image(target_image, (args.image_size, args.image_size))
        target_image = normalize_image(target_image, args.min_hu, args.max_hu)
        target_image = torch.tensor(target_image, dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        if args.degradation_type == "ldct":
            samples, labels, ldct_severities, lact_severities, svct_severities, alphas, degradations = generate_low_dose_samples(target_image)

        elif args.degradation_type == "lact":
            samples, labels, ldct_severities, lact_severities, svct_severities, end_angles, degradations = generate_limited_angle_samples(target_image)

        elif args.degradation_type == "svct":
            samples, labels, ldct_severities, lact_severities, svct_severities, views, degradations = generate_sparse_view_samples(target_image)

        # Save the simulated sinograms and reconstructions
        for i, sample in enumerate(samples):
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            sample_path = os.path.join(output_image_dir, f"{base_filename}_sample_{i}.npy")
            np.save(sample_path, sample.cpu().numpy())
            label_path = os.path.join(output_image_dir, f"{base_filename}_label_{i}.npy")
            np.save(label_path, labels[i].cpu().numpy())
            dataset_info["image_paths"].append(sample_path)
            dataset_info["label_paths"].append(label_path)
            dataset_info["window_centers"].append([(window_low + window_high) / 2])
            dataset_info["window_widths"].append([window_high - window_low])
            dataset_info["ldct_severities"].append(ldct_severities[i])
            dataset_info["lact_severities"].append(lact_severities[i])
            dataset_info["svct_severities"].append(svct_severities[i])
            dataset_info["degradations"].append(degradations[i])
        
        if args.degradation_type == "ldct":
            dataset_info["alphas"].extend(alphas)
            dataset_info["views"].extend([None] * len(alphas))
            dataset_info["end_angles"].extend([None] * len(alphas))
        elif args.degradation_type == "lact":
            dataset_info["end_angles"].extend(end_angles)
            dataset_info["alphas"].extend([None] * len(end_angles))
            dataset_info["views"].extend([None] * len(end_angles))
        elif args.degradation_type == "svct":
            dataset_info["views"].extend(views)
            dataset_info["alphas"].extend([None] * len(views))
            dataset_info["end_angles"].extend([None] * len(views))
            
    # Save dataset info
    info_path = os.path.join(args.output_dir, "dataset_info.csv")
    pd.DataFrame(dataset_info).to_csv(info_path, index=False)
    print(f"Dataset info saved to {info_path}")

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    main()