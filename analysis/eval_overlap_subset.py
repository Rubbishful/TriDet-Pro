"""
重叠场景子集评估（使用内置 ANETdetection.evaluate_overlap_subset）
"""
import sys, os, argparse, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.utils.metrics import ANETdetection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json', required=True)
    parser.add_argument('--pred_pkl', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--overlap_iou', type=float, default=0.3)
    args = parser.parse_args()

    with open(args.pred_pkl, 'rb') as f:
        preds = pickle.load(f)

    det_eval = ANETdetection(args.gt_json, split=args.split,
                             tiou_thresholds=np.linspace(0.3, 0.7, 5))

    results = det_eval.evaluate_overlap_subset(preds, overlap_tiou=args.overlap_iou, verbose=True)

if __name__ == '__main__':
    main()
