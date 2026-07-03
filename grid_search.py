"""快速搜索最优 NMS 超参数"""
import sys, os
sys.path.insert(0, '.')

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed
import torch, glob, itertools, json

cfg = load_config('./configs/thumos_i3d.yaml')
val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
val_loader = make_data_loader(val_dataset, False, None, 1, 0)

# 加载模型
model = make_meta_arch(cfg['model_name'], **cfg['model'])
model = torch.nn.DataParallel(model, device_ids=[0])
ckpt_dir = './ckpt/thumos_i3d_thumos_baseline'
ckpt_file = sorted(glob.glob(f'{ckpt_dir}/*.pth.tar'))[-1]
ckpt = torch.load(ckpt_file, map_location='cuda:0')
model.load_state_dict(ckpt['state_dict_ema'])
model.cuda().eval()

# 参数空间
sigmas = [0.55, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.65]
iou_ts = [0.08, 0.10, 0.12, 0.15]
pre_topk = [3000, 5000, 7000]

best = (0, {})
# 只搜 sigma (最关键参数)
for sigma in sigmas:
    import copy
    c = copy.deepcopy(cfg)
    c['test_cfg']['nms_sigma'] = sigma
    c['test_cfg']['pre_nms_topk'] = 5000
    c['test_cfg']['max_seg_num'] = 3000

    # 临时修改 model 属性
    for k, v in c['test_cfg'].items():
        if hasattr(model.module, f'test_{k}'):
            setattr(model.module, f'test_{k}', v)
    model.module.test_nms_sigma = sigma

    det_eval = ANETdetection(val_dataset.json_file, val_dataset.split[0],
                              tiou_thresholds=torch.linspace(0.3, 0.7, 5))
    mAP = valid_one_epoch(val_loader, model, -1, evaluator=det_eval,
                           output_file=None, ext_score_file=None,
                           tb_writer=None, print_freq=999)
    print(f'sigma={sigma:.2f}  mAP={mAP*100:.2f}%')
    if mAP > best[0]:
        best = (mAP, {'sigma': sigma})

print(f'\nBest: mAP={best[0]*100:.2f}%  config={best[1]}')
