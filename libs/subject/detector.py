import numpy as np
from ultralytics import YOLO


class SubjectDetector:
    """封装 YOLO 检测器，用于从视频帧中检测人物主体。

    Args:
        model_name: YOLO 模型文件名或路径
        device: 推理设备 ('cuda', 'cpu', 'mps')
    """

    def __init__(self, model_name='yolov8n.pt', device='cuda'):
        self.model = YOLO(model_name)
        self.device = device

    def detect(self, frame: np.ndarray, conf_threshold=0.5) -> list[dict]:
        """对单帧图像做人物检测。

        Args:
            frame: (H, W, 3) BGR 图像
            conf_threshold: 置信度阈值

        Returns:
            list[dict]: 检测结果列表, 每项为 {'bbox': [x1,y1,x2,y2], 'confidence': float}
        """
        results = self.model(frame, device=self.device, classes=[0], conf=conf_threshold, verbose=False)
        persons = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                persons.append({'bbox': box.tolist(), 'confidence': float(conf)})
        return persons

    def detect_at_timestamp(self, video_path: str, timestamp: float) -> list[dict]:
        """在视频指定时间戳检测人物。

        Args:
            video_path: 视频文件路径
            timestamp: 时间戳（秒）

        Returns:
            list[dict]: 检测结果列表
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return []
        return self.detect(frame)
