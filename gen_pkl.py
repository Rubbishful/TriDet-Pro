"""极简：直接调用 valid_one_epoch 存 pkl"""
import os, sys, argparse
sys.argv = ['gen_pkl.py']

# reuse eval.py's main but force args
class Args:
    config = './configs/thumos_i3d.yaml'
    ckpt = './ckpt/thumos_i3d_thumos_baseline/'
    saveonly = True
    topk = -1
    print_freq = 20

# import eval module
exec(open('eval.py').read())
main(Args())
