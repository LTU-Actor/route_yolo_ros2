from ultralytics import YOLO
import numpy as np
import torch
import gc
import cv2
import easyocr 

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, UInt32
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from route_yolo_service.srv import DetectObject



class YoloDetector(Node):
    def __init__(self):
        super().__init__('YoloDetector')
        self.bridge = CvBridge()
        self.cam_image = None
        self.reader = easyocr.Reader(["en"], gpu=True, verbose=False)

        # Declare parameters
        self.declare_parameter("image_topic", "/routecam/image_raw")
        self.declare_parameter("coco_model_path", "")
        self.declare_parameter("tire_model_path", "")
        self.declare_parameter('flip_image', False)
        self.declare_parameter('image_resize', 640)

        # Load YOLO model paths
        self.model_coco_path = self.get_parameter("coco_model_path").value
        self.model_tire_path = self.get_parameter("tire_model_path").value

        # Subscribers
        self.create_subscription(ROSImage, self.get_parameter("image_topic").value, self.image_callback, 10)

        # Publishers
        self.detection_window_pub = self.create_publisher(ROSImage, "yolo_detection_window", 1)

        # Service
        self.create_service(DetectObject, 'detect_object', self.handle_detection_request)

        self.get_logger().info("YOLO Service-Based Detector Initialized")

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CVBridge error: {e}")
            return

        resize_val = self.get_parameter('image_resize').get_parameter_value().integer_value
        if resize_val > 0:
            img = self.letterbox_resize(img, (resize_val, resize_val))

        if self.get_parameter('flip_image').get_parameter_value().bool_value:
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)

        self.cam_image = img

    def handle_detection_request(self, request, response):
        if self.cam_image is None:
            self.get_logger().warn("No image received yet")
            response.count = 0
            response.size = 0
            return response

        if request.target not in ['stop', 'tire', 'person']:
            self.get_logger().warn(f"Unknown detection target: {request.target}")
            response.count = 0
            response.size = 0
            return response

        count, size = self.detect(request.target)
        response.count = count
        response.size = size
        return response

    def detect(self, mode):
        model_path = {
            'stop': self.model_coco_path,
            'tire': self.model_tire_path,
            'person': self.model_coco_path
        }[mode]

        im = self.cam_image.copy()
        yolo = YOLO(model_path)
        
        target_ids = {
            'person': 0,
            'stop': 11,
            'tire': 0,
        }

        results = yolo.predict(source=im, device="0", stream=False, verbose=False, conf=0.5, classes=[target_ids[mode]], show=False)

        count, biggest = self.analyze_results(results, im, mode)
        self.cam_image = None
        torch.cuda.empty_cache()
        gc.collect()
        
        return count, biggest

    def analyze_results(self, results, image : cv2.Mat, mode):
        detected = 0
        biggest_bbox = 0
        image_size = image.shape[0] * image.shape[1]
        output_image = image.copy()
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                xyxy = box.xyxy
                if mode == 'stop':
                    sign_image = image[int(xyxy[0][1]):int(xyxy[0][3]), int(xyxy[0][0]):int(xyxy[0][2])]
                    if not self.stopsign_ocr_check(sign_image):
                        cv2.line(output_image, (int(xyxy[0][0]),int(xyxy[0][1])), (int(xyxy[0][2]),int(xyxy[0][3])), (0,0,255), 3)
                        cv2.putText(output_image, (f"Fake {mode}, {round(box.conf.item(), 2)}"), (int(xyxy[0][0]), int(xyxy[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        continue

                detected += 1
                area = 100 * ((box.xywh[0][2] * box.xywh[0][3]) / image_size)
                if area > biggest_bbox:
                    biggest_bbox = area

                cv2.rectangle(output_image, (int(xyxy[0][0]),int(xyxy[0][1])), (int(xyxy[0][2]),int(xyxy[0][3])), (0,0,255), 2)
                cv2.putText(output_image, (f"{mode}, {round(box.conf.item(), 2)}"), (int(xyxy[0][0]), int(xyxy[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    

        self.detection_window_pub.publish(self.bridge.cv2_to_imgmsg(output_image, "bgr8"))
        return detected, int(biggest_bbox)
    
    def stopsign_ocr_check(self, sign_image):
        stop_found = False
        readings = self.reader.readtext(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
        for _, text, _ in readings:
            text.replace("0", "O")
            self.get_logger().info(f"Sign reads: {text}")
            if "STOP" in text.upper():
                stop_found = True
                break
        return stop_found
    

    def letterbox_resize(self, img, size=(640, 640)):
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        if h == w:
            return cv2.resize(img, size, cv2.INTER_AREA)
        dif = max(h, w)
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        y, x = (dif - h) // 2, (dif - w) // 2
        mask[y:y+h, x:x+w] = img
        return cv2.resize(mask, size)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

