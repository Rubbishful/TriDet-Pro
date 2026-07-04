import os
import sys
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2

from libs.subject.detector import SubjectDetector

VIDEO_PATH = 'test_video.mp4'


def draw_detections(frame, persons):
    """在帧上绘制检测到的人像边界框和置信度。"""
    for p in persons:
        x1, y1, x2, y2 = map(int, p['bbox'])
        conf = p['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'person {conf:.2f}', (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"错误：找不到视频文件 '{VIDEO_PATH}'")
        print("请将 test_video.mp4 放到 project_test/ 目录下后重试。")
        sys.exit(1)

    print(f"正在加载 YOLO 模型...")
    detector = SubjectDetector(device='cpu')
    print("模型加载完成。")

    print(f"正在打开视频: {VIDEO_PATH} ...")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{VIDEO_PATH}'，文件可能已损坏。")
        sys.exit(1)

    print("视频已打开，开始检测... (按 'q' 键退出)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束。")
            break

        persons = detector.detect(frame, conf_threshold=0.5)
        draw_detections(frame, persons)

        cv2.imshow('Subject Detection - YOLOv8', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("程序正常结束。")


if __name__ == '__main__':
    main()
