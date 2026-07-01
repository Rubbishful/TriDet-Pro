"""最小化推理脚本：加载模型 → 跑 212 视频 → 存 pkl"""
import os, sys, time, pickle, glob
import torch
import torch.nn as nn

# 1. 加载配置
from libs.core import load_config
cfg = load_config('./configs/thumos_i3d.yaml')

# 2. 加载数据集
from libs.datasets import make_dataset, make_data_loader
val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
val_loader = make_data_loader(val_dataset, False, None, 1, 0)

# 3. 加载模型
from libs.modeling import make_meta_arch
model = make_meta_arch(cfg['model_name'], **cfg['model'])
model = nn.DataParallel(model, device_ids=[torch.device('cuda:0')])

ckpt_dir = './ckpt/thumos_i3d_thumos_baseline'
ckpt_list = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth.tar')))
ckpt = torch.load(ckpt_list[-1], map_location='cuda:0')
model.load_state_dict(ckpt['state_dict_ema'])
model.eval()

# 4. 推理
results = {'version': 'V0'}
total = len(val_loader)
t0 = time.time()
for i, video_list in enumerate(val_loader):
    with torch.no_grad():
        output = model(video_list)
    for vid_idx, vid_name in enumerate(video_list['video_name']):
        results[vid_name] = {
            'segments': output[vid_idx]['segments'].cpu(),
            'scores': output[vid_idx]['scores'].cpu(),
            'labels': output[vid_idx]['labels'].cpu(),
            'duration': video_list['duration'][vid_idx],
            'feat_stride': video_list['feat_stride'][vid_idx],
            'feat_num_frames': video_list['feat_num_frames'][vid_idx],
        }
    if (i + 1) % 20 == 0:
        print(f'[{i+1}/{total}] {time.time()-t0:.1f}s')

# 5. 保存
out = os.path.join(ckpt_dir, 'eval_results.pkl')
with open(out, 'wb') as f:
    pickle.dump(results, f)
print(f'Done! Saved to {out} ({time.time()-t0:.1f}s)')
