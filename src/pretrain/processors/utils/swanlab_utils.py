# utils/swanlab_utils.py
import swanlab
import torch
from torchvision.utils import make_grid
from utils.utils import get_adaptive_window

def config_swanlab(
    project_name: str,
    run_name: str,
    config: dict | None = None,
    logdir: str | None = None,
    mode: str = "cloud",
    description: str | None = None,
):
    """
    初始化 SwanLab 实验。

    Args:
        project_name: SwanLab 项目名
        run_name: 本次实验的名字
        config: 超参数等配置（会显示在 Config 面板）
        logdir: 本地日志保存目录（可选，None 则用默认）
        mode: "cloud" 或 "local"
        description: 实验描述
    """
    swanlab.login(api_key="bkDTFgqsuqKw1jxfuf9lU")
    run = swanlab.init(
        project=project_name,
        experiment_name=run_name,
        config=config,
        logdir=logdir,
        mode=mode,
        description=description,
    )
    return run


def log_metrics_to_swanlab(
    metrics: dict,
    mode: str = "train",
    step: int | None = None,
):
    """
    记录标量指标到 SwanLab。

    metrics: 例如 {"loss": 0.1, "psnr": 30.2, "ssim": 0.9}
    mode: "train" / "val" / "test" 等，用作前缀
    """
    log_dict = {f"{mode}/{k}": float(v) for k, v in metrics.items()}
    swanlab.log(log_dict, step=step)


# def log_images_to_swanlab(
#     inputs: torch.Tensor,
#     targets: torch.Tensor,
#     outputs: torch.Tensor,
#     mode: str = "train",
#     step: int | None = None,
#     max_images: int = 4,
# ):
#     """
#     将若干张图像以 batch 形式上传到 SwanLab。

#     inputs / targets / outputs:
#         形状 [B, C, H, W] 或 [B, 1, H, W] 的 torch.Tensor
#     """
#     if inputs.ndim != 4 or targets.ndim != 4 or outputs.ndim != 4:
#         raise ValueError("inputs/targets/outputs 必须是 [B, C, H, W] 形状的 4D Tensor")

#     b = min(max_images, inputs.size(0))

#     imgs_dict = {
#         f"{mode}/input": swanlab.Image(
#             inputs[:b].detach().cpu(), caption=f"{mode} inputs (first {b})"
#         ),
#         f"{mode}/target": swanlab.Image(
#             targets[:b].detach().cpu(), caption=f"{mode} targets (first {b})"
#         ),
#         f"{mode}/output": swanlab.Image(
#             outputs[:b].detach().cpu(), caption=f"{mode} outputs (first {b})"
#         ),
#     }
#     swanlab.log(imgs_dict, step=step)


# def log_images_to_swanlab(
#     inputs: torch.Tensor,
#     targets: torch.Tensor,
#     outputs: torch.Tensor,
#     mode: str = "train",
#     step: int | None = None,
#     max_images: int = 4,
#     make_triplet_grid: bool = True,
# ):
#     """
#     inputs / targets / outputs: [B, C, H, W] 的 torch.Tensor

#     如果 make_triplet_grid=True:
#         - 生成一个大 grid：
#           第 1 行: degraded
#           第 2 行: output
#           第 3 行: target
#         - 一行里有若干个样本，方便对比

#     如果 make_triplet_grid=False:
#         - 分别记录三组 batch（和你之前的逻辑类似）
#     """
#     if inputs.ndim != 4 or targets.ndim != 4 or outputs.ndim != 4:
#         raise ValueError("inputs/targets/outputs 必须是 [B, C, H, W] 形状的 4D Tensor")

#     b = min(max_images, inputs.size(0))

#     if make_triplet_grid:
#         # 取前 b 张
#         in_b = inputs[:b].detach().cpu()
#         out_b = outputs[:b].detach().cpu()
#         tgt_b = targets[:b].detach().cpu()

#         # 拼成 [3B, C, H, W]：先所有 input，再所有 output，再所有 target
#         stacked = torch.cat([in_b, out_b, tgt_b], dim=0)  # (3b, C, H, W)

#         # 用 make_grid 拼成一张大图
#         # nrow=b => 每行放 b 张图片，所以 3b 张图会变成 3 行：
#         #   第1行: b张 input
#         #   第2行: b张 output
#         #   第3行: b张 target
#         grid = make_grid(
#             stacked,
#             nrow=b,
#             padding=2,
#             normalize=False,  # 如果你这里传的是归一化到[0,1]的，就保持 False；如果是 HU，并想自动拉伸可以 True
#         )  # 结果: [C, H_grid, W_grid]

#         caption = (
#             f"{mode} grid: row1=input, row2=output, row3=target; "
#             f"showing {b} samples"
#         )
#         swanlab.log(
#             {
#                 f"{mode}/grid_triplet": swanlab.Image(
#                     grid,
#                     caption=caption,
#                     # size=512,  # 可选：限制最长边，避免太大
#                 )
#             },
#             step=step,
#         )
#     else:
#         # 退回到“分 3 张图记录”的老逻辑
#         in_b = inputs[:b].detach().cpu()
#         out_b = outputs[:b].detach().cpu()
#         tgt_b = targets[:b].detach().cpu()

#         imgs_dict = {
#             f"{mode}/input": swanlab.Image(in_b, caption=f"{mode} inputs (first {b})"),
#             f"{mode}/target": swanlab.Image(tgt_b, caption=f"{mode} targets (first {b})"),
#             f"{mode}/output": swanlab.Image(out_b, caption=f"{mode} outputs (first {b})"),
#         }
#         swanlab.log(imgs_dict, step=step)


def log_images_to_swanlab(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    outputs: torch.Tensor,
    mode: str = "train",
    step: int | None = None,
    max_images: int = 4,
    use_sample_window: bool = True,
    min_hu: float = -1024.0,
    max_hu: float = 1024.0,
):
    """
    inputs / targets / outputs: [B, C, H, W]，建议是 HU 域。

    功能：
      - 对于 batch 中每个 sample：
          (input_i, output_i, target_i) 三张图共用一个 window；
      - 所有样本一起画成 3 行 × b 列的 grid：
          第 1 行: degraded (input)
          第 2 行: output (model)
          第 3 行: target (clean)
    """
    if inputs.ndim != 4 or targets.ndim != 4 or outputs.ndim != 4:
        raise ValueError("inputs/targets/outputs 必须是 [B, C, H, W] 形状的 4D Tensor")

    # 只取前 max_images 个样本
    b = min(max_images, inputs.size(0))

    # 搬到 CPU
    in_b = inputs[:b].detach().cpu()
    out_b = outputs[:b].detach().cpu()
    tgt_b = targets[:b].detach().cpu()

    inputs_norm = []
    outputs_norm = []
    targets_norm = []

    for i in range(b):
        # 第 i 个 sample: [C, H, W]
        in_i = in_b[i]
        out_i = out_b[i]
        tgt_i = tgt_b[i]

        if use_sample_window:
            # 在这个 sample 的 3 张图上一起估计 window
            triple = torch.stack([in_i, out_i, tgt_i], dim=0)  # [3, C, H, W]
            vmin, vmax = get_adaptive_window(
                triple,
                percentile_low=1.0,
                percentile_high=99.0,
                min_hu=min_hu,
                max_hu=max_hu,
                central_fraction=0.9,
                min_window=200.0,
            )
            triple = torch.clamp(triple, vmin, vmax)
            triple = (triple - vmin) / (vmax - vmin + 1e-8)  # -> [0,1]

            in_i_n = triple[0]
            out_i_n = triple[1]
            tgt_i_n = triple[2]
        else:
            # 不做 window，假设外面已经是 [0,1]
            in_i_n = in_i
            out_i_n = out_i
            tgt_i_n = tgt_i

        inputs_norm.append(in_i_n)
        outputs_norm.append(out_i_n)
        targets_norm.append(tgt_i_n)

    inputs_norm = torch.stack(inputs_norm, dim=0)   # [B, C, H, W]
    outputs_norm = torch.stack(outputs_norm, dim=0) # [B, C, H, W]
    targets_norm = torch.stack(targets_norm, dim=0) # [B, C, H, W]

    # 拼成 [3B, C, H, W]，顺序：所有 input -> 所有 output -> 所有 target
    all_imgs = torch.cat([inputs_norm, outputs_norm, targets_norm], dim=0)

    # nrow = b => 3 行 × b 列：
    #   第 1 行: inputs_norm[0..b-1]
    #   第 2 行: outputs_norm[0..b-1]
    #   第 3 行: targets_norm[0..b-1]
    grid = make_grid(
        all_imgs,
        nrow=b,
        padding=2,
        normalize=False,  # 我们已经把它弄到 [0,1] 了
    )

    caption = (
        f"{mode} grid (per-sample window): "
        f"row1=input (degraded), row2=output (model), row3=target (clean); "
        f"showing {b} samples."
    )

    swanlab.log(
        {
            f"{mode}/grid_triplet": swanlab.Image(
                grid,
                caption=caption,
            )
        },
        step=step,
    )
