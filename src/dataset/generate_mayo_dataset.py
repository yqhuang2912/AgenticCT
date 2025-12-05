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
from utils import load_dicom_image, get_adaptive_window, denormalize_image, normalize_image, resize_image

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


LDCT_SEVERITY = {
    "high": 0.2,
    "medium": 0.4,
    "low": 0.6
}

SVCT_SEVERITY = {
    "high": 60,
    "medium": 120,
    "low": 300
}

LACT_SEVERITY = {
    "high": 90,
    "medium": 120,
    "low": 150
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

    # single degradation (3)
    for severity in SVCT_SEVERITY.keys():
        n_views = SVCT_SEVERITY[severity]
        every_n_views = ANGLES // n_views
        p_svct = simulate_sparse_view(p_fdct, every_n_views)
        svct_options = create_sparse_view_geometry(every_n_views)
        svct_reconstruction = fbp(p_svct, svct_options)
        samples.append(svct_reconstruction)
        labels.append(input_image)
        ldct_severities.append(None)
        svct_severities.append(severity)
        lact_severities.append(None)
        views.append(n_views)
        degradations.append(['svct'])

    # two degradations
    # 1) ldct + svct (9)
    for ldct_severity in LDCT_SEVERITY.keys():
        alpha = LDCT_SEVERITY[ldct_severity]
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        label = fbp(p_ldct, BASE_OPTIONS)
        for svct_severity in SVCT_SEVERITY.keys():
            n_views = SVCT_SEVERITY[svct_severity]
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

    # 2) lact + svct (9)
    for lact_severity in LACT_SEVERITY.keys():
        end_angle_deg = LACT_SEVERITY[lact_severity]
        p_lact = simulate_limited_angle(p_fdct, end_angle_deg)
        lact_options = create_limited_angle_geometry(end_angle_deg)
        label = fbp(p_lact, lact_options)
        for svct_severity in SVCT_SEVERITY.keys():
            n_views = SVCT_SEVERITY[svct_severity]
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

    # three degradations (27)
    for ldct_severity in LDCT_SEVERITY.keys():
        alpha = LDCT_SEVERITY[ldct_severity]
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        for lact_severity in LACT_SEVERITY.keys():
            end_angle_deg = LACT_SEVERITY[lact_severity]
            p_lact = simulate_limited_angle(p_ldct, end_angle_deg)
            lact_options = create_limited_angle_geometry(end_angle_deg)
            label = fbp(p_lact, lact_options)
            for svct_severity in SVCT_SEVERITY.keys():
                n_views = SVCT_SEVERITY[svct_severity]
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

    # single degradation (3)
    for severity in LACT_SEVERITY.keys():
        end_angle_deg = LACT_SEVERITY[severity]
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
    # ldct + lact (9)
    for ldct_severity in LDCT_SEVERITY.keys():
        alpha = LDCT_SEVERITY[ldct_severity]
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        label = fbp(p_ldct, BASE_OPTIONS)
        for lact_severity in LACT_SEVERITY.keys():
            end_angle_deg = LACT_SEVERITY[lact_severity]
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

    # lact + svct (9)
    for lact_severity in LACT_SEVERITY.keys():
        end_angle_deg = LACT_SEVERITY[lact_severity]
        for svct_severity in SVCT_SEVERITY.keys():
            n_views = SVCT_SEVERITY[svct_severity]
            every_n_views = ANGLES // n_views

            p_lasvct, view_interval = simulate_sparse_limited_angle(
                p_fdct, every_n_views, end_angle_deg
            )
            lasvct_options = create_sparse_limited_angle_geometry(view_interval)
            lasvct_reconstruction = fbp(p_lasvct, lasvct_options)
            samples.append(lasvct_reconstruction)

            p_svct = simulate_sparse_view(p_fdct, every_n_views)
            svct_options = create_sparse_view_geometry(every_n_views)
            label = fbp(p_svct, svct_options)
            labels.append(label)

            lact_severities.append(lact_severity)
            svct_severities.append(svct_severity)
            ldct_severities.append(None)
            end_angles.append(end_angle_deg)
            degradations.append(['lact', 'svct'])

    # three degradations (27)
    for ldct_severity in LDCT_SEVERITY.keys():
        alpha = LDCT_SEVERITY[ldct_severity]
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        for lact_severity in LACT_SEVERITY.keys():
            end_angle_deg = LACT_SEVERITY[lact_severity]
            for svct_severity in SVCT_SEVERITY.keys():
                n_views = SVCT_SEVERITY[svct_severity]
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

    # single degradation (3)
    for severity in LDCT_SEVERITY.keys():
        alpha = LDCT_SEVERITY[severity]
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        ldct_reconstruction = fbp(p_ldct, BASE_OPTIONS)
        samples.append(ldct_reconstruction)
        labels.append(input_image)
        ldct_severities.append(severity)
        svct_severities.append(None)
        lact_severities.append(None)
        alphas.append(alpha)
        degradations.append(['ldct'])

    # two degradations
    # ldct + svct (9)
    for ldct_severity in LDCT_SEVERITY.keys():
        alpha = LDCT_SEVERITY[ldct_severity]
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        for svct_severity in SVCT_SEVERITY.keys():
            n_views = SVCT_SEVERITY[svct_severity]
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

    # ldct + lact (9)
    for ldct_severity in LDCT_SEVERITY.keys():
        alpha = LDCT_SEVERITY[ldct_severity]
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        for lact_severity in LACT_SEVERITY.keys():
            end_angle_deg = LACT_SEVERITY[lact_severity]
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

    # three degradations (27)
    for ldct_severity in LDCT_SEVERITY.keys():
        alpha = LDCT_SEVERITY[ldct_severity]
        p_ldct = simulate_low_dose_gaussian(p_fdct, alpha)
        for lact_severity in LACT_SEVERITY.keys():
            end_angle_deg = LACT_SEVERITY[lact_severity]
            for svct_severity in SVCT_SEVERITY.keys():
                n_views = SVCT_SEVERITY[svct_severity]
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
    parser.add_argument("--target_image_dir", type=str, default="/data/hyq/data/mayo", help="Path to the input CT image")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the simulated sinograms and reconstructions")
    parser.add_argument("--image_size", type=int, default=256, help="Size to resize the input image (default: 256)")
    parser.add_argument("--min_hu", type=int, default=-1024, help="Minimum HU value for normalization (default: -1024)")
    parser.add_argument("--max_hu", type=int, default=3072, help="Maximum HU value for normalization (default: 3072)")
    parser.add_argument("--slice_thickness", type=str, choices=["1mm", "3mm"], default="3mm", help="Slice thickness in mm")
    parser.add_argument("--exclude_patient_ids", type=str, choices=PATIENT_IDS, default=[], help="Patient IDs to exclude from the dataset")
    parser.add_argument("--degradation_type", type=str, choices=["ldct", "lact", "svct"], required=True, help="Type of degradation to simulate")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for 8:1:1 split")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset_info = {
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

    val_dataset_info = {
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

    test_dataset_info = {
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

    all_targets = []
    for pid in PATIENT_IDS:
        if pid in args.exclude_patient_ids:
            continue
        base_dir = os.path.join(args.target_image_dir, pid, f"full_{args.slice_thickness}")
        if not os.path.exists(base_dir):
            continue
        paths = sorted(glob.glob(os.path.join(base_dir, "*.IMA")))
        for p in paths:
            all_targets.append((pid, p))

    rng = random.Random(args.split_seed)
    rng.shuffle(all_targets)
    N = len(all_targets)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    train_list = all_targets[:n_train]
    val_list = all_targets[n_train:n_train + n_val]
    test_list = all_targets[n_train + n_val:]

    def process_entries(entries, dataset_info, desc):
        for pid, target_image_path in tqdm(entries, desc=desc):
            output_patient_dir = os.path.join(args.output_dir, pid, f"full_{args.slice_thickness}")
            os.makedirs(output_patient_dir, exist_ok=True)
            target_image, window_center, window_width = load_dicom_image(target_image_path)
            target_image = resize_image(target_image, (args.image_size, args.image_size))
            target_image = normalize_image(target_image, args.min_hu, args.max_hu)
            target_image = torch.tensor(target_image, dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
            if args.degradation_type == "ldct":
                samples, labels, ldct_severities, lact_severities, svct_severities, alphas, degradations = generate_low_dose_samples(target_image)
            elif args.degradation_type == "lact":
                samples, labels, ldct_severities, lact_severities, svct_severities, end_angles, degradations = generate_limited_angle_samples(target_image)
            elif args.degradation_type == "svct":
                samples, labels, ldct_severities, lact_severities, svct_severities, views, degradations = generate_sparse_view_samples(target_image)
            for i, sample in enumerate(samples):
                base_filename = os.path.splitext(os.path.basename(target_image_path))[0]
                sample_path = os.path.join(output_patient_dir, f"{base_filename}_sample_{i}.npy")
                np.save(sample_path, sample.cpu().numpy())
                label_path = os.path.join(output_patient_dir, f"{base_filename}_label_{i}.npy")
                np.save(label_path, labels[i].cpu().numpy())
                dataset_info["image_paths"].append(sample_path)
                dataset_info["label_paths"].append(label_path)
                dataset_info["window_centers"].append(window_center)
                dataset_info["window_widths"].append(window_width)
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

    process_entries(train_list, train_dataset_info, "Processing train")
    process_entries(val_list, val_dataset_info, "Processing val")
    process_entries(test_list, test_dataset_info, "Processing test")

    train_info_path = os.path.join(args.output_dir, "train_dataset_info.csv")
    pd.DataFrame(train_dataset_info).to_csv(train_info_path, index=False)
    print(f"Train dataset info saved to {train_info_path}")
    val_info_path = os.path.join(args.output_dir, "val_dataset_info.csv")
    pd.DataFrame(val_dataset_info).to_csv(val_info_path, index=False)
    print(f"Val dataset info saved to {val_info_path}")
    test_info_path = os.path.join(args.output_dir, "test_dataset_info.csv")
    pd.DataFrame(test_dataset_info).to_csv(test_info_path, index=False)
    print(f"Test dataset info saved to {test_info_path}")

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
    # import pydicom
    # import matplotlib.pyplot as plt

    

    # mayo2016_filepath = '/data/hyq/data/mayo/L109/full_1mm/L109_FD_1_1.CT.0001.0100.2015.12.23.17.52.25.829117.125774352.IMA'
    # min_hu = -1024
    # max_hu = 3072
    # mayo2016_image, wc, ww = load_dicom_image(mayo2016_filepath)
    # vmin = wc[0] - ww[0] / 2
    # vmax = wc[0] + ww[0] / 2
    # image_size = 256

    # every_n_views = ANGLES // 100  # 每隔多少个视角采样一次
    # end_angle_deg = 240

    # import cv2
    # mayo2016_image = cv2.resize(mayo2016_image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    # print("max and min of original image:", mayo2016_image.max(), mayo2016_image.min())
    # mayo2016_image = normalize_image(mayo2016_image, min_hu, max_hu)
    # mayo2016_image = torch.tensor(mayo2016_image, dtype=torch.float32, device='cuda')
    # mayo2016_image = mayo2016_image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    

    # p_fdct = projection(mayo2016_image, BASE_OPTIONS)

    # low_dose_poisson = simulate_low_dose_poisson(p_fdct, alpha=0.25)
    # low_dose_gaussian = simulate_low_dose_gaussian(p_fdct, alpha=0.25)
    # sparse_view = simulate_sparse_view(p_fdct, every_n_views=every_n_views)
    # limited_angle = simulate_limited_angle(p_fdct, end_angle_deg=end_angle_deg)
    # sparse_limited_angle, view_indices = simulate_sparse_limited_angle(p_fdct, every_n_views=every_n_views, end_angle_deg=end_angle_deg)
    # low_dose_sparse_view = simulate_sparse_view(low_dose_gaussian, every_n_views=every_n_views)
    # limited_angle_low_dose = simulate_limited_angle(low_dose_gaussian, end_angle_deg=end_angle_deg)
    # low_dose_sparse_limited_angle, fuse_view_indices = simulate_sparse_limited_angle(low_dose_gaussian, every_n_views=every_n_views, end_angle_deg=end_angle_deg)


    # low_dose_reconstruction_poisson = fbp(low_dose_poisson, BASE_OPTIONS)
    # low_dose_reconstruction_gaussian = fbp(low_dose_gaussian, BASE_OPTIONS)

    # sparse_view_options = create_sparse_view_geometry(every_n_views)
    # sparse_view_reconstruction = fbp(sparse_view, sparse_view_options)
    # low_dose_sparse_view_reconstruction = fbp(low_dose_sparse_view, sparse_view_options)

    # limited_angle_options = create_limited_angle_geometry(end_angle_deg)
    # limited_angle_reconstruction = fbp(limited_angle, limited_angle_options)
    # limited_angle_low_dose_reconstruction = fbp(limited_angle_low_dose, limited_angle_options)

    # sparse_limited_angle_options = create_sparse_limited_angle_geometry(view_indices)
    # sparse_limited_angle_reconstruction = fbp(sparse_limited_angle, sparse_limited_angle_options)

    # low_dose_sparse_limited_angle_reconstruction = fbp(low_dose_sparse_limited_angle, sparse_limited_angle_options)

    # low_dose_reconstruction_gaussian = denormalize_image(low_dose_reconstruction_gaussian, min_hu, max_hu)
    # low_dose_reconstruction_poisson = denormalize_image(low_dose_reconstruction_poisson, min_hu, max_hu)
    # sparse_view_reconstruction = denormalize_image(sparse_view_reconstruction, min_hu, max_hu)
    # low_dose_sparse_view_reconstruction = denormalize_image(low_dose_sparse_view_reconstruction, min_hu, max_hu)
    # limited_angle_reconstruction = denormalize_image(limited_angle_reconstruction, min_hu, max_hu)
    # limited_angle_low_dose_reconstruction = denormalize_image(limited_angle_low_dose_reconstruction, min_hu, max_hu)
    # sparse_limited_angle_reconstruction = denormalize_image(sparse_limited_angle_reconstruction, min_hu, max_hu)
    # low_dose_sparse_limited_angle_reconstruction = denormalize_image(low_dose_sparse_limited_angle_reconstruction, min_hu, max_hu)
    # mayo2016_image = denormalize_image(mayo2016_image, min_hu, max_hu)

    # print("max and min of original image:", torch.max(mayo2016_image), torch.min(mayo2016_image))
    # print("max and min of low dose reconstruction (poisson):", torch.max(low_dose_reconstruction_poisson), torch.min(low_dose_reconstruction_poisson))
    # print("max and min of low dose reconstruction (gaussian):", torch.max(low_dose_reconstruction_gaussian), torch.min(low_dose_reconstruction_gaussian))
    # print("max and min of sparse view reconstruction:", torch.max(sparse_view_reconstruction), torch.min(sparse_view_reconstruction))
    # print("max and min of low dose sparse view reconstruction:", torch.max(low_dose_sparse_view_reconstruction), torch.min(low_dose_sparse_view_reconstruction))
    # print("max and min of limited angle reconstruction:", torch.max(limited_angle_reconstruction), torch.min(limited_angle_reconstruction))
    # print("max and min of limited angle low dose reconstruction:", torch.max(limited_angle_low_dose_reconstruction), torch.min(limited_angle_low_dose_reconstruction))
    # print("max and min of sparse limited angle reconstruction:", torch.max(sparse_limited_angle_reconstruction), torch.min(sparse_limited_angle_reconstruction))
    # print("max and min of low dose sparse limited angle reconstruction:", torch.max(low_dose_sparse_limited_angle_reconstruction), torch.min(low_dose_sparse_limited_angle_reconstruction))

    # plt.figure(figsize=(8, 36))
    # plt.subplot(9, 2, 1)
    # plt.title('Original Image')
    # plt.imshow(mayo2016_image.squeeze().cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    # plt.axis('off')

    # plt.subplot(9, 2, 2)
    # plt.title('Sinogram')
    # plt.imshow(p_fdct.squeeze().cpu().numpy(), cmap='gray', aspect='auto')
    # plt.axis('off')

    # plt.subplot(9, 2, 3)
    # plt.title('Low-Dose FBP Reconstruction (Poisson)')
    # plt.imshow(low_dose_reconstruction_poisson.squeeze().cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    # plt.axis('off')

    # plt.subplot(9, 2, 4)
    # plt.title('Low-Dose Sinogram (Poisson)')
    # plt.imshow(low_dose_poisson.squeeze().cpu().numpy(), cmap='gray', aspect='auto')
    # plt.axis('off')

    # plt.subplot(9, 2, 5)
    # plt.title('Low-Dose FBP Reconstruction (Gaussian)')
    # plt.imshow(low_dose_reconstruction_gaussian.squeeze().cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    # plt.axis('off')

    # plt.subplot(9, 2, 6)
    # plt.title('Low-Dose Sinogram (Gaussian)')
    # plt.imshow(low_dose_gaussian.squeeze().cpu().numpy(), cmap='gray', aspect='auto')
    # plt.axis('off')

    # sv_vmin, sv_vmax = get_adaptive_window(sparse_view_reconstruction.cpu())
    # plt.subplot(9, 2, 7)
    # plt.title('Sparse-View FBP Reconstruction')
    # plt.imshow(sparse_view_reconstruction.squeeze().cpu().numpy(), cmap='gray', vmin=sv_vmin, vmax=sv_vmax)
    # plt.axis('off')

    # plt.subplot(9, 2, 8)
    # plt.title('Sparse-View Sinogram')
    # plt.imshow(sparse_view.squeeze().cpu().numpy(), cmap='gray', aspect='auto')
    # plt.axis('off')

    # la_vmin, la_vmax = get_adaptive_window(limited_angle_reconstruction.cpu())
    # plt.subplot(9, 2, 9)
    # plt.title('Limited-Angle FBP Reconstruction')
    # plt.imshow(limited_angle_reconstruction.squeeze().cpu().numpy(), cmap='gray', vmin=la_vmin, vmax=la_vmax)
    # plt.axis('off')

    # plt.subplot(9, 2, 10)
    # plt.title('Limited-Angle Sinogram')
    # plt.imshow(limited_angle.squeeze().cpu().numpy(), cmap='gray', aspect='auto')
    # plt.axis('off')

    # sla_vmin, sla_vmax = get_adaptive_window(sparse_limited_angle_reconstruction.cpu())
    # plt.subplot(9, 2, 11)
    # plt.title('Sparse & Limited-Angle FBP Reconstruction')
    # plt.imshow(sparse_limited_angle_reconstruction.squeeze().cpu().numpy(), cmap='gray', vmin=sla_vmin, vmax=sla_vmax)
    # plt.axis('off') 

    # plt.subplot(9, 2, 12)
    # plt.title('Sparse & Limited-Angle Sinogram')
    # plt.imshow(sparse_limited_angle.squeeze().cpu().numpy(), cmap='gray', aspect='auto')
    # plt.axis('off')

    # ldsla_vmin, ldsla_vmax = get_adaptive_window(low_dose_sparse_limited_angle_reconstruction.cpu())
    # plt.subplot(9, 2, 13)
    # plt.title('Low-Dose Sparse & Limited-Angle FBP Reconstruction')
    # plt.imshow(low_dose_sparse_limited_angle_reconstruction.squeeze().cpu().numpy(), cmap='gray', vmin=ldsla_vmin, vmax=ldsla_vmax)
    # plt.axis('off')

    # plt.subplot(9, 2, 14)
    # plt.title('Low-Dose Sparse & Limited-Angle Sinogram')
    # plt.imshow(low_dose_sparse_limited_angle.squeeze().cpu().numpy(), cmap='gray', aspect='auto')
    # plt.axis('off')

    # ldsv_vmin, ldsv_vmax = get_adaptive_window(low_dose_sparse_view_reconstruction.cpu())
    # plt.subplot(9, 2, 15)
    # plt.title('Low-Dose Sparse-View FBP Reconstruction')
    # plt.imshow(low_dose_sparse_view_reconstruction.squeeze().cpu().numpy(), cmap='gray', vmin=ldsv_vmin, vmax=ldsv_vmax)
    # plt.axis('off')

    # plt.subplot(9, 2, 16)
    # plt.title('Low-Dose Sparse-View Sinogram')
    # plt.imshow(low_dose_sparse_view.squeeze().cpu().numpy(), cmap='gray', aspect='auto')
    # plt.axis('off') 

    # ldla_vmin, ldla_vmax = get_adaptive_window(limited_angle_low_dose_reconstruction.cpu())
    # plt.subplot(9, 2, 17)
    # plt.title('Limited-Angle Low-Dose FBP Reconstruction')
    # plt.imshow(limited_angle_low_dose_reconstruction.squeeze().cpu().numpy(), cmap='gray', vmin=ldla_vmin, vmax=ldla_vmax)
    # plt.axis('off')

    # plt.subplot(9, 2, 18)
    # plt.title('Limited-Angle Low-Dose Sinogram')
    # plt.imshow(limited_angle_low_dose.squeeze().cpu().numpy(), cmap='gray', aspect='auto')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.savefig('low_dose_simulation.png', dpi=300)

