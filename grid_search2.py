"""扩展搜索：iou_threshold + pre_nms_thresh + 不同 epoch"""
import sys, os, copy, glob
sys.path.insert(0, '.')

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection
import torch, numpy as np

cfg = load_config('./configs/thumos_i3d.yaml')
val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
val_loader = make_data_loader(val_dataset, False, None, 1, 0)
det_eval = ANETdetection(val_dataset.json_file, val_dataset.split[0],
                          tiou_thresholds=np.linspace(0.3, 0.7, 5))

results = []

# 1. 搜索 iou_threshold (NMS IoU)
ckpt_dir = './ckpt/thumos_i3d_thumos_baseline'
ckpt_file = sorted(glob.glob(f'{ckpt_dir}/*.pth.tar'))[-1]
model = make_meta_arch(cfg['model_name'], **cfg['model'])
model = torch.nn.DataParallel(model, device_ids=[0])
ckpt = torch.load(ckpt_file, map_location='cuda:0')
model.load_state_dict(ckpt['state_dict_ema'])
model.cuda().eval()

for iou_t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    model.module.test_iou_threshold = iou_t
    model.module.test_nms_sigma = 0.60
    model.module.test_pre_nms_topk = 5000
    model.module.test_max_seg_num = 3000
    mAP = valid_one_epoch(val_loader, model, -1, evaluator=det_eval,
                           output_file=None, ext_score_file=None,
                           tb_writer=None, print_freq=999)
    print(f'iou_thresh={iou_t:.2f}  mAP={mAP*100:.2f}%')
    results.append((mAP, f'iou_thresh={iou_t}'))

# 2. 搜索 pre_nms_thresh
for pnt in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
    model.module.test_pre_nms_thresh = pnt
    model.module.test_iou_threshold = 0.10
    mAP = valid_one_epoch(val_loader, model, -1, evaluator=det_eval,
                           output_file=None, ext_score_file=None,
                           tb_writer=None, print_freq=999)
    print(f'pre_nms_thresh={pnt:.4f}  mAP={mAP*100:.2f}%')
    results.append((mAP, f'pre_nms_thresh={pnt}'))

# 3. 搜索不同 epoch
for ep in range(25, 40, 5):
    ep_file = f'{ckpt_dir}/epoch_{ep:03d}.pth.tar'
    if not os.path.exists(ep_file): continue
    ckpt2 = torch.load(ep_file, map_location='cuda:0')
    model.load_state_dict(ckpt2['state_dict_ema'])
    model.module.test_pre_nms_thresh = 0.001
    model.module.test_iou_threshold = 0.10
    model.module.test_nms_sigma = 0.60
    mAP = valid_one_epoch(val_loader, model, -1, evaluator=det_eval,
                           output_file=None, ext_score_file=None,
                           tb_writer=None, print_freq=999)
    print(f'epoch={ep}  mAP={mAP*100:.2f}%')
    results.append((mAP, f'epoch={ep}'))

best = max(results, key=lambda x: x[0])
print(f'\nBest: {best[1]}  mAP={best[0]*100:.2f}%')
