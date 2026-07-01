# Functions for 1D NMS, modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
import os, sys

# Ensure this directory is on sys.path so the compiled .pyd can be found
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import torch

# [CHANGED] 将 PyTorch lib 目录加入 Windows DLL 搜索路径，
# 否则 nms_1d_cpu.pyd 在导入时找不到 c10.dll / torch_cpu.dll
import struct
if struct.calcsize("P") == 8:
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.isdir(_torch_lib) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_torch_lib)

# Try to import the C extension, fall back to pure PyTorch
try:
    import nms_1d_cpu
    _has_nms_cpu = True
except ImportError:
    _has_nms_cpu = False


def _nms_1d_torch(segs, scores, iou_threshold):
    """Pure PyTorch implementation of 1D NMS."""
    if segs.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=segs.device)

    x1 = segs[:, 0]
    x2 = segs[:, 1]
    areas = x2 - x1 + 1e-6

    _, order = scores.sort(0, descending=True)
    keep = torch.ones(segs.size(0), dtype=torch.bool, device=segs.device)

    for _i in range(segs.size(0)):
        if not keep[_i]:
            continue
        i = order[_i]
        ix1, ix2, iarea = x1[i], x2[i], areas[i]

        for _j in range(_i + 1, segs.size(0)):
            if not keep[_j]:
                continue
            j = order[_j]
            inter = max(0.0, min(ix2, x2[j]) - max(ix1, x1[j]))
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= iou_threshold:
                keep[_j] = False

    return order[keep]


def _softnms_1d_torch(segs, scores, iou_threshold, sigma, min_score, method):
    """Pure PyTorch implementation of 1D soft NMS.

    method: 0=hard NMS, 1=linear, 2=gaussian
    Returns indices of kept segments and the dets tensor (N x 3: x1, x2, score).
    """
    nsegs = segs.size(0)
    if nsegs == 0:
        return torch.empty(0, dtype=torch.long, device=segs.device), segs.new_empty((0, 3))

    x1 = segs[:, 0].clone()
    x2 = segs[:, 1].clone()
    sc = scores.clone()
    areas = x2 - x1 + 1e-6
    inds = torch.arange(nsegs, dtype=torch.long, device=segs.device)

    dets = segs.new_empty((nsegs, 3))
    pos = 0

    while pos < nsegs:
        # find max score among remaining
        remaining_scores = sc[pos:nsegs]
        max_idx = remaining_scores.argmax()
        max_pos = pos + max_idx.item()

        # swap max_pos <-> pos
        ix1 = dets[pos, 0] = x1[max_pos].item()
        ix2 = dets[pos, 1] = x2[max_pos].item()
        iscore = dets[pos, 2] = sc[max_pos].item()
        iarea = areas[max_pos].item()
        iind = inds[max_pos].item()

        x1[max_pos], x1[pos] = x1[pos].item(), ix1
        x2[max_pos], x2[pos] = x2[pos].item(), ix2
        sc[max_pos], sc[pos] = sc[pos].item(), iscore
        areas[max_pos], areas[pos] = areas[pos].item(), iarea
        inds[max_pos], inds[pos] = inds[pos].item(), iind

        # go through remaining
        j = pos + 1
        while j < nsegs:
            xx1 = max(ix1, x1[j].item())
            xx2 = min(ix2, x2[j].item())
            inter = max(0.0, xx2 - xx1)
            ovr = inter / (iarea + areas[j].item() - inter)

            if method == 0:       # hard nms
                weight = 0.0 if ovr >= iou_threshold else 1.0
            elif method == 1:     # linear
                weight = (1.0 - ovr) if ovr >= iou_threshold else 1.0
            else:                 # gaussian
                weight = float(torch.exp(torch.tensor(-(ovr * ovr) / sigma)))

            sc[j] = sc[j].item() * weight

            if sc[j] < min_score:
                # swap with last
                nsegs -= 1
                if nsegs == j:
                    break
                x1[j], x1[nsegs] = x1[nsegs].item(), x1[j].item()
                x2[j], x2[nsegs] = x2[nsegs].item(), x2[j].item()
                sc[j], sc[nsegs] = sc[nsegs].item(), sc[j].item()
                areas[j], areas[nsegs] = areas[nsegs].item(), areas[j].item()
                inds[j], inds[nsegs] = inds[nsegs].item(), inds[j].item()
            else:
                j += 1
        pos += 1

    return inds[:nsegs], dets[:pos]


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, min_score, max_num
    ):
        # vanilla nms will not change the score, so we can filter segs first
        is_filtering_by_score = (min_score > 0)
        if is_filtering_by_score:
            valid_mask = scores > min_score
            segs, scores = segs[valid_mask], scores[valid_mask]
            cls_idxs = cls_idxs[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        # nms op; return inds that is sorted by descending order
        if _has_nms_cpu:
            inds = nms_1d_cpu.nms(
                segs.contiguous().cpu(),
                scores.contiguous().cpu(),
                iou_threshold=float(iou_threshold))
        else:
            inds = _nms_1d_torch(segs, scores, iou_threshold)
        # cap by max number
        if max_num > 0:
            inds = inds[:min(max_num, len(inds))]
        # return the sorted segs / scores
        sorted_segs = segs[inds]
        sorted_scores = scores[inds]
        sorted_cls_idxs = cls_idxs[inds]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


# Track which backend is being used (diagnostic, printed once)
_nms_backend_checked = False

class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, sigma, min_score, method, max_num
    ):
        global _nms_backend_checked
        if not _nms_backend_checked:
            _nms_backend_checked = True
            if _has_nms_cpu:
                print(f"[NMS] Using C extension backend (nms_1d_cpu)")
            else:
                print(f"[NMS] Using pure PyTorch fallback (slower)")
        if _has_nms_cpu:
            # pre allocate memory for sorted results
            dets = segs.new_empty((segs.size(0), 3), device='cpu')
            inds = nms_1d_cpu.softnms(
                segs.cpu(),
                scores.cpu(),
                dets.cpu(),
                iou_threshold=float(iou_threshold),
                sigma=float(sigma),
                min_score=float(min_score),
                method=int(method))
            # cap by max number
            if max_num > 0:
                n_segs = min(len(inds), max_num)
            else:
                n_segs = len(inds)
            sorted_segs = dets[:n_segs, :2]
            sorted_scores = dets[:n_segs, 2]
            sorted_cls_idxs = cls_idxs[inds]
            sorted_cls_idxs = sorted_cls_idxs[:n_segs]
        else:
            inds, dets = _softnms_1d_torch(
                segs.cpu(), scores.cpu(), iou_threshold, sigma, min_score, method)
            if max_num > 0:
                n_segs = min(len(inds), max_num)
            else:
                n_segs = len(inds)
            sorted_segs = dets[:n_segs, :2]
            sorted_scores = dets[:n_segs, 2]
            sorted_cls_idxs = cls_idxs[inds[:n_segs]]

        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


def compute_local_density(segs, iou_threshold=0.3):
    """
    Compute local prediction density for each segment.
    Returns a vector of shape (N,) where each entry is the fraction of
    other segments that overlap this segment with tIoU > iou_threshold.

    Args:
        segs (Tensor): N x 2 tensor of (start, end)
        iou_threshold (float): tIoU threshold for "overlapping"

    Returns:
        Tensor: (N,) density scores in [0, 1]
    """
    n = segs.shape[0]
    if n <= 1:
        return segs.new_zeros(n)

    x1 = segs[:, 0]
    x2 = segs[:, 1]
    areas = x2 - x1 + 1e-6

    # pairwise intersection
    left = torch.maximum(x1[:, None], x1[None, :])
    right = torch.minimum(x2[:, None], x2[None, :])
    inter = (right - left).clamp(min=0)

    # pairwise iou
    union = areas[:, None] + areas[None, :] - inter
    iou = inter / union

    # density = fraction of other segs overlapping this one
    overlap_mask = (iou > iou_threshold).float()
    # exclude self
    overlap_mask.fill_diagonal_(0.0)
    density = overlap_mask.sum(dim=1) / (n - 1)

    return density


def _cross_class_dedup(segs, scores, labels, iou_thresh=0.7, penalty=0.9):
    """ [CHANGED] 向量化实现，替代原先 O(N^2) Python 双循环

    Cross-class de-duplication: for predictions that overlap heavily but
    have different class labels, apply a small score penalty to the
    lower-scored prediction. This prevents the same temporal segment
    from being detected as multiple different action classes.
    """
    n = segs.shape[0]
    if n <= 1:
        return scores

    # 向量化 pairwise IoU
    x1 = segs[:, 0]
    x2 = segs[:, 1]
    areas = x2 - x1 + 1e-6

    left = torch.maximum(x1[:, None], x1[None, :])
    right = torch.minimum(x2[:, None], x2[None, :])
    inter = (right - left).clamp(min=0)
    union = areas[:, None] + areas[None, :] - inter
    iou = inter / union

    # 只惩罚不同类别、IoU超过阈值的低分预测
    diff_class = (labels[:, None] != labels[None, :])
    overlap = iou > iou_thresh
    # i 分比 j 高 → 惩罚 j；triu 只取上三角避免重复
    mask = torch.triu(diff_class & overlap, diagonal=1)

    lower_scored = scores[:, None] < scores[None, :]
    to_penalize = mask & lower_scored

    penalty_factors = to_penalize.float() * (penalty - 1.0) + 1.0
    penalty_per_item = penalty_factors.min(dim=1).values

    return scores * penalty_per_item


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    """
        blur localization results by incorporating side segs.
        this is known as bounding box voting in object detection literature.
        slightly boost the performance around iou_threshold
    """

    # *_segs : N_i x 2, all_scores: N,
    # apply offset
    offset_scores = all_scores + score_offset

    # computer overlap between nms and all segs
    # construct the distance matrix of # N_nms x # N_all
    num_nms_segs, num_all_segs = nms_segs.shape[0], all_segs.shape[0]
    ex_nms_segs = nms_segs[:, None].expand(num_nms_segs, num_all_segs, 2)
    ex_all_segs = all_segs[None, :].expand(num_nms_segs, num_all_segs, 2)

    # compute intersection
    left = torch.maximum(ex_nms_segs[:, :, 0], ex_all_segs[:, :, 0])
    right = torch.minimum(ex_nms_segs[:, :, 1], ex_all_segs[:, :, 1])
    inter = (right-left).clamp(min=0)

    # lens of all segments
    nms_seg_lens = ex_nms_segs[:, :, 1] - ex_nms_segs[:, :, 0]
    all_seg_lens = ex_all_segs[:, :, 1] - ex_all_segs[:, :, 0]

    # iou
    iou = inter / (nms_seg_lens + all_seg_lens - inter)

    # get neighbors (# N_nms x # N_all) / weights
    seg_weights = (iou >= iou_threshold).to(all_scores.dtype) * all_scores[None, :]
    seg_weights /= torch.sum(seg_weights, dim=1, keepdim=True)
    refined_segs = seg_weights @ all_segs

    return refined_segs

def batched_nms(
    segs,
    scores,
    cls_idxs,
    iou_threshold,
    min_score,
    max_seg_num,
    use_soft_nms=True,
    multiclass=True,
    sigma=0.5,
    voting_thresh=0.75,
    # Scheme A: Density-Aware NMS
    adaptive_nms=False,
    density_iou_thresh=0.3,
    beta=0.15,
    sigma_high_density=0.85,
    density_threshold=0.3,
    # Scheme B: Cross-class dedup + score calibration
    cross_class_dedup=False,
    cross_class_iou_thresh=0.7,
    cross_class_penalty=0.9,
    density_score_penalty=False,
):
    # Based on Detectron2 implementation,
    num_segs = segs.shape[0]
    # corner case, no prediction outputs
    if num_segs == 0:
        return torch.zeros([0, 2]),\
               torch.zeros([0,]),\
               torch.zeros([0,], dtype=cls_idxs.dtype)

    if multiclass:
        # multiclass nms: apply nms on each class independently
        new_segs, new_scores, new_cls_idxs = [], [], []
        for class_id in torch.unique(cls_idxs):
            curr_indices = torch.where(cls_idxs == class_id)[0]
            class_segs = segs[curr_indices]
            class_scores = scores[curr_indices]
            class_cls = cls_idxs[curr_indices]

            # ---- Scheme A: Density-Aware NMS (统一NMS + 密集提分) ----
            if adaptive_nms:
                density = compute_local_density(class_segs, density_iou_thresh)
                is_dense = density > density_threshold
                if is_dense.any():
                    # [CHANGED] 不拆分两组，而是统一 NMS + 密集区域提分
                    # 密集区域的预测得分上调，使它们在 NMS 中更早被处理、更不容易被抑制
                    score_boost = 1.0 + beta * density.clamp(max=1.0)
                    boosted_scores = class_scores * score_boost
                    sorted_segs, sorted_scores, sorted_cls_idxs = SoftNMSop.apply(
                        class_segs, boosted_scores, class_cls,
                        iou_threshold,
                        sigma_high_density,  # [CHANGED] 统一使用高 sigma，无跨组丢失
                        min_score,
                        2,
                        max_seg_num
                    )
                    # 将提升后的得分 clamp 回 [0, 1]
                    sorted_scores = sorted_scores.clamp(max=1.0)
                else:
                    sorted_segs, sorted_scores, sorted_cls_idxs = SoftNMSop.apply(
                        class_segs, class_scores, class_cls,
                        iou_threshold,
                        sigma,
                        min_score,
                        2,
                        max_seg_num
                    )
            # ---- End Scheme A ----
            elif use_soft_nms:
                sorted_segs, sorted_scores, sorted_cls_idxs = SoftNMSop.apply(
                    class_segs,
                    class_scores,
                    class_cls,
                    iou_threshold,
                    sigma,
                    min_score,
                    2,
                    max_seg_num
                )
            else:
                sorted_segs, sorted_scores, sorted_cls_idxs = NMSop.apply(
                    class_segs,
                    class_scores,
                    class_cls,
                    iou_threshold,
                    min_score,
                    max_seg_num
                )
            # disable seg voting for multiclass nms, no sufficient segs

            # fill in the class index
            new_segs.append(sorted_segs)
            new_scores.append(sorted_scores)
            new_cls_idxs.append(sorted_cls_idxs)

        # cat the results
        new_segs = torch.cat(new_segs)
        new_scores = torch.cat(new_scores)
        new_cls_idxs = torch.cat(new_cls_idxs)

        # ---- Scheme B: Cross-class de-duplication ----
        if cross_class_dedup:
            new_scores = _cross_class_dedup(
                new_segs, new_scores, new_cls_idxs,
                iou_thresh=cross_class_iou_thresh,
                penalty=cross_class_penalty
            )

        # ---- Scheme B: Density score penalty ----
        if density_score_penalty:
            # For predictions in dense regions, slightly reduce scores
            # to favour diversity — prevents a single instance from
            # dominating multiple overlapping GT matches
            density = compute_local_density(new_segs, density_iou_thresh)
            # Apply mild penalty: score *= (1 - 0.1 * density)
            penalty_factor = 1.0 - 0.1 * density
            new_scores = new_scores * penalty_factor.clamp(min=0.7)

    else:
        # class agnostic
        if adaptive_nms:
            density = compute_local_density(segs, density_iou_thresh)
            is_dense = density > density_threshold
            if is_dense.any():
                # [CHANGED] 统一 NMS + 密集提分，不拆分
                score_boost = 1.0 + beta * density.clamp(max=1.0)
                boosted_scores = scores * score_boost
                new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
                    segs, boosted_scores, cls_idxs,
                    iou_threshold,
                    sigma_high_density,
                    min_score,
                    2,
                    max_seg_num
                )
                new_scores = new_scores.clamp(max=1.0)
            else:
                new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
                    segs, scores, cls_idxs, iou_threshold,
                    sigma, min_score, 2, max_seg_num
                )
        elif use_soft_nms:
            new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                sigma, min_score, 2, max_seg_num
            )
        else:
            new_segs, new_scores, new_cls_idxs = NMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                min_score, max_seg_num
            )
        # seg voting
        if voting_thresh > 0:
            new_segs = seg_voting(
                new_segs,
                segs,
                scores,
                voting_thresh
            )

    # sort based on scores and return
    # truncate the results based on max_seg_num
    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    # needed for multiclass NMS
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs
