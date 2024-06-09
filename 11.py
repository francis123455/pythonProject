import os
import tarfile
import urllib.request
import numpy as np
import tensorflow as tf
# import cv2


# 下载SSD MobileNet V2模型
MODEL_NAME = 'ssd_mobilenet_v2'
MODEL_DATE = '20200711'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)
if not os.path.exists(MODEL_DIR):
    MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/' + MODEL_DATE + '/' + MODEL_TAR_FILENAME
    urllib.request.urlretrieve(MODEL_URL, MODEL_TAR_FILENAME)
    tar = tarfile.open(MODEL_TAR_FILENAME)
    tar.extractall(MODELS_DIR)
    tar.close()
    os.remove(MODEL_TAR_FILENAME)

# 加载模型
model_dir = os.path.join('models/ssd_mobilenet_v2', 'saved_model')
detect_fn = tf.saved_model.load(model_dir)

# video_url = 'https://www.bilibili.com/video/BV1EU4y1b7Dd/?spm_id_from=333.999.0.0'
video_path = 'SnapAny.mp4'
# urllib.request.urlretrieve(video_url, video_path)

# SORT跟踪算法实现
class Sort:
    def __init__(self):
        self.trackers = []
        self.track_id_count = 0

    def update(self, detections, frame):
        updated_tracks = []

        # 更新已有的跟踪器
        for tracker, track_id in self.trackers:
            success, box = tracker.update(frame)
            if success:
                updated_tracks.append((box, track_id))
            else:
                self.trackers.remove((tracker, track_id))

        # 初始化新的跟踪器
        for det in detections:
            x, y, w, h = det
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, tuple(det))
            self.trackers.append((tracker, self.track_id_count))
            updated_tracks.append((det, self.track_id_count))
            self.track_id_count += 1

        return updated_tracks

sort_tracker = Sort()

def apply_nms(boxes, scores, iou_threshold=0.6):
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=50, iou_threshold=iou_threshold)
    return selected_indices.numpy()

# 打开视频文件
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为TensorFlow模型需要的输入格式
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)

    # 进行目标检测
    detections = detect_fn(input_tensor)

    # 解析检测结果
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    classes = detections['detection_classes'].astype(np.int64)

    # 只保留置信度高的“person”检测结果
    confidence_threshold = 0.65 # 可以根据需要调整
    indices = np.where((scores >= confidence_threshold) & (classes == 1))[0]
    boxes = boxes[indices]
    scores = scores[indices]

    # 应用非极大值抑制（NMS）
    if len(boxes) > 0:
        nms_indices = apply_nms(boxes, scores)
        boxes = boxes[nms_indices]
        scores = scores[nms_indices]

    # 转换为xywh格式
    detections = []
    h, w, _ = frame.shape
    for box in boxes:
        y1, x1, y2, x2 = box
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        detections.append([x1, y1, x2 - x1, y2 - y1])

    # 更新跟踪器
    tracks = sort_tracker.update(detections, frame)
    if len(tracks) > 0:
        tracks = [tracks[-1]]
    else:
        tracks = tracks
    # 绘制检测和跟踪结果
    for bbox, _ in tracks:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # 显示结果
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()