import torch
import math


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

# 参考陈阳论文
LDCT_MIN, LDCT_MAX = 0.05, 0.75
LACT_MIN, LACT_MAX = 75, 270
SVCT_MIN, SVCT_MAX = 15, 360

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