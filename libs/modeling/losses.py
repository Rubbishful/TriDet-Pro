import torch
from torch.nn import functional as F


@torch.jit.script
def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_giou_loss_1d(
        input_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        reduction: str = 'none',
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # giou is reduced to iou in our setting, skip unnecessary steps
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_diou_loss_1d(
        input_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        reduction: str = 'none',
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_eiou_loss_1d(
        input_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        reduction: str = 'none',
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Efficient IoU Loss (Zhang et. al)
    https://arxiv.org/abs/2101.08158

    1D simplification:
      L_EIoU_1D = 1 - IoU + (rho^2 + (w_pred - w_gt)^2) / len_c^2
    where rho is the center offset, len_c is the enclosing segment width,
    w_pred/w_gt are the predicted/GT segment widths.

    In 1D, c = wc = len_c (the enclosing box IS the single dimension),
    so the EIoU penalty simplifies to a distance term (same as DIoU)
    plus a width penalty term that explicitly penalizes width mismatch.

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    rho = 0.5 * (rp - lp - rg + lg)

    # width penalty: directly minimizes width difference
    w_pred = lp + rp
    w_gt = lg + rg
    width_diff = w_pred - w_gt

    # eiou: L_IoU + L_dist + L_width
    loss = 1.0 - iouk + (torch.square(rho) + torch.square(width_diff)) / torch.square(len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_alpha_diou_loss_1d(
        input_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        reduction: str = 'none',
        eps: float = 1e-8,
        alpha: float = 3.0,
) -> torch.Tensor:
    """
    Alpha-DIoU Loss: Power IoU Loss for Bounding Box Regression (He et. al)
    https://arxiv.org/abs/2110.13675

    1D formula:
      L_Alpha-DIoU = (1 - IoU + rho^2 / len_c^2)^alpha

    With alpha > 1, the loss up-weights high IoU samples (well-predicted
    positives), improving regression accuracy at stricter IoU thresholds.
    Paper default alpha = 3 performs well across most settings.

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
        eps (float): small number to prevent division by zero
        alpha (float): power parameter, > 1 up-weights high IoU samples
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    rho = 0.5 * (rp - lp - rg + lg)

    # diou per-sample
    diou = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    # power transform
    loss = torch.pow(diou, alpha)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_focaler_diou_loss_1d(
        input_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        reduction: str = 'none',
        eps: float = 1e-8,
        d: float = 0.0,
        u: float = 0.95,
) -> torch.Tensor:
    """
    Focaler-DIoU Loss (Zhang et. al)
    https://arxiv.org/abs/2401.10525

    1D formula:
      IoU_focaler = { 0               if IoU < d
                    { (IoU-d)/(u-d)   if d <= IoU <= u
                    { 1               if IoU > u
      L_Focaler-DIoU = 1 - IoU_focaler + rho^2 / len_c^2

    Equivalent to: L_DIoU + (IoU - IoU_focaler)
    This re-weights samples based on IoU quality, focusing gradients on
    the [d, u] interval. Samples with IoU < d are ignored (hard negatives),
    and samples with IoU > u are saturated (near-perfect matches).

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
        eps (float): small number to prevent division by zero
        d (float): lower IoU threshold, samples below are zeroed
        u (float): upper IoU threshold, samples above are saturated
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    rho = 0.5 * (rp - lp - rg + lg)

    # Focaler-IoU: piecewise linear re-mapping
    iou_focaler = torch.where(
        iouk < d,
        torch.zeros_like(iouk),
        torch.where(
            iouk > u,
            torch.ones_like(iouk),
            (iouk - d) / max(u - d, eps),
        ),
    )

    # L_Focaler-DIoU = 1 - IoU_focaler + distance penalty
    loss = 1.0 - iou_focaler + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
