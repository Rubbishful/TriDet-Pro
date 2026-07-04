from __future__ import annotations

import os
import numpy as np
from ultralytics import YOLO

_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SubjectDetector:
    def __init__(self, model_name=None, device='cpu'):
        if model_name is None:
            model_name = os.path.join(_MODEL_DIR, 'yolov8n.pt')
        self.model = YOLO(model_name)
        self.device = device

    def detect(self, frame: np.ndarray, conf_threshold=0.5) -> list[dict]:
        """
        对单帧图像进行人物检测。
        """
        results = self.model(frame, device=self.device, verbose=False)
        persons = []
        
        # 遍历检测结果
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                # COCO数据集中，'person' 的类别ID是 0
                if cls == 0:  
                    conf = float(boxes.conf[i].item())
                    # 过滤掉置信度低的误检
                    if conf > conf_threshold:
                        # 提取边界框坐标 [x1, y1, x2, y2]
                        bbox = boxes.xyxy[i].cpu().numpy().tolist()
                        persons.append({
                            'bbox': bbox,
                            'confidence': round(conf, 2)
                        })
        return persons