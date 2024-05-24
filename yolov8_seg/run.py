import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
import tf2_ros
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray, Marker

import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from copy import copy
import time

from rknnlite.api import RKNNLite



RK3588_RKNN_MODEL = 'yolov8_seg.rknn'
IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

OBJ_THRESH = 0.50
NMS_THRESH = 0.45
MAX_DETECT = 300

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class Letter_Box_Info():
    def __init__(self, shape, new_shape, w_ratio, h_ratio, dw, dh, pad_color) -> None:
        self.origin_shape = shape
        self.new_shape = new_shape
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.dw = dw 
        self.dh = dh
        self.pad_color = pad_color

class COCO_test_helper():
    def __init__(self, enable_letter_box = False) -> None:
        self.record_list = []
        self.enable_ltter_box = enable_letter_box
        if self.enable_ltter_box is True:
            self.letter_box_info_list = []
        else:
            self.letter_box_info_list = None

    def letter_box(self, im, new_shape, pad_color=(0,0,0), info_need=False):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border
        
        if self.enable_ltter_box is True:
            self.letter_box_info_list.append(Letter_Box_Info(shape, new_shape, ratio, ratio, dw, dh, pad_color))
        if info_need is True:
            return im, ratio, (dw, dh)
        else:
            return im

    def direct_resize(self, im, new_shape, info_need=False):
        shape = im.shape[:2]
        h_ratio = new_shape[0]/ shape[0]
        w_ratio = new_shape[1]/ shape[1]
        if self.enable_ltter_box is True:
            self.letter_box_info_list.append(Letter_Box_Info(shape, new_shape, w_ratio, h_ratio, 0, 0, (0,0,0)))
        im = cv2.resize(im, (new_shape[1], new_shape[0]))
        return im

    def get_real_box(self, box, in_format='xyxy'):
        bbox = copy(box)
        if self.enable_ltter_box == True:
        # unletter_box result
            if in_format=='xyxy':
                bbox[:,0] -= self.letter_box_info_list[-1].dw
                bbox[:,0] /= self.letter_box_info_list[-1].w_ratio
                bbox[:,0] = np.clip(bbox[:,0], 0, self.letter_box_info_list[-1].origin_shape[1])

                bbox[:,1] -= self.letter_box_info_list[-1].dh
                bbox[:,1] /= self.letter_box_info_list[-1].h_ratio
                bbox[:,1] = np.clip(bbox[:,1], 0, self.letter_box_info_list[-1].origin_shape[0])

                bbox[:,2] -= self.letter_box_info_list[-1].dw
                bbox[:,2] /= self.letter_box_info_list[-1].w_ratio
                bbox[:,2] = np.clip(bbox[:,2], 0, self.letter_box_info_list[-1].origin_shape[1])

                bbox[:,3] -= self.letter_box_info_list[-1].dh
                bbox[:,3] /= self.letter_box_info_list[-1].h_ratio
                bbox[:,3] = np.clip(bbox[:,3], 0, self.letter_box_info_list[-1].origin_shape[0])
        return bbox

    def get_real_seg(self, seg):
        #! fix side effect
        dh = int(self.letter_box_info_list[-1].dh)
        dw = int(self.letter_box_info_list[-1].dw)
        origin_shape = self.letter_box_info_list[-1].origin_shape
        new_shape = self.letter_box_info_list[-1].new_shape
        if (dh == 0) and (dw == 0) and origin_shape == new_shape:
            return seg
        elif dh == 0 and dw != 0:
            seg = seg[:, :, dw:-dw] # a[0:-0] = []
        elif dw == 0 and dh != 0 : 
            seg = seg[:, dh:-dh, :]
        seg = np.where(seg, 1, 0).astype(np.uint8).transpose(1,2,0)
        seg = cv2.resize(seg, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_LINEAR)
        if len(seg.shape) < 3:
            return seg[None,:,:]
        else:
            return seg.transpose(2,0,1)

    def add_single_record(self, image_id, category_id, bbox, score, in_format='xyxy', pred_masks = None):
        if self.enable_ltter_box == True:
        # unletter_box result
            if in_format=='xyxy':
                bbox[0] -= self.letter_box_info_list[-1].dw
                bbox[0] /= self.letter_box_info_list[-1].w_ratio

                bbox[1] -= self.letter_box_info_list[-1].dh
                bbox[1] /= self.letter_box_info_list[-1].h_ratio

                bbox[2] -= self.letter_box_info_list[-1].dw
                bbox[2] /= self.letter_box_info_list[-1].w_ratio

                bbox[3] -= self.letter_box_info_list[-1].dh
                bbox[3] /= self.letter_box_info_list[-1].h_ratio
                # bbox = [value/self.letter_box_info_list[-1].ratio for value in bbox]

        if in_format=='xyxy':
        # change xyxy to xywh
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
        else:
            assert False, "now only support xyxy format, please add code to support others format"
        
        def single_encode(x):
            from pycocotools.mask import encode
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        if pred_masks is None:
            self.record_list.append({"image_id": image_id,
                                    "category_id": category_id,
                                    "bbox":[round(x, 3) for x in bbox],
                                    'score': round(score, 5),
                                    })
        else:
            rles = single_encode(pred_masks)
            self.record_list.append({"image_id": image_id,
                                    "category_id": category_id,
                                    "bbox":[round(x, 3) for x in bbox],
                                    'score': round(score, 5),
                                    'segmentation': rles,
                                    })
    

def dfl(position):
    # Distribution Focal Loss (DFL)
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()

def _crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
    
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0] //grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def filter_boxes(boxes, box_confidences, box_class_probs, seg_part):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

    return boxes, classes, scores, seg_part

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def post_process(input_data):
    # input_data[0], input_data[4], and input_data[8] are detection box information
    # input_data[1], input_data[5], and input_data[9] are category score information
    # input_data[2], input_data[6], and input_data[10] are confidence score information
    # input_data[3], input_data[7], and input_data[11] are segmentation information
    # input_data[12] is the proto information
    proto = input_data[-1]
    boxes, scores, classes_conf, seg_part = [], [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))
        seg_part.append(input_data[pair_per_branch*i+3])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    seg_part = [sp_flatten(_v) for _v in seg_part]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    seg_part = np.concatenate(seg_part)

    # filter according to threshold
    boxes, classes, scores, seg_part = filter_boxes(boxes, scores, classes_conf, seg_part)

    zipped = zip(boxes, classes, scores, seg_part)
    sort_zipped = sorted(zipped, key=lambda x: (x[2]), reverse=True)
    result = zip(*sort_zipped)

    max_nms = 30000
    n = boxes.shape[0]  # number of boxes
    if not n:
        return None, None, None, None
    elif n > max_nms:  # excess boxes
        boxes, classes, scores, seg_part = [np.array(x[:max_nms]) for x in result]
    else:
        boxes, classes, scores, seg_part = [np.array(x) for x in result]

    # nms
    nboxes, nclasses, nscores, nseg_part = [], [], [], []
    agnostic = 0
    max_wh = 7680
    c = classes * (0 if agnostic else max_wh)
    ids = torchvision.ops.nms(torch.tensor(boxes, dtype=torch.float32) + torch.tensor(c, dtype=torch.float32).unsqueeze(-1),
                              torch.tensor(scores, dtype=torch.float32), NMS_THRESH)
    real_keeps = ids.tolist()[:MAX_DETECT]
    nboxes.append(boxes[real_keeps])
    nclasses.append(classes[real_keeps])
    nscores.append(scores[real_keeps])
    nseg_part.append(seg_part[real_keeps])

    if not nclasses and not nscores:
        return None, None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    seg_part = np.concatenate(nseg_part)

    ph, pw = proto.shape[-2:]
    proto = proto.reshape(seg_part.shape[-1], -1)
    seg_img = np.matmul(seg_part, proto)
    seg_img = sigmoid(seg_img)
    seg_img = seg_img.reshape(-1, ph, pw)

    seg_threadhold = 0.5

    # crop seg outside box
    seg_img = F.interpolate(torch.tensor(seg_img)[None], torch.Size([640, 640]), mode='bilinear', align_corners=False)[0]
    seg_img_t = _crop_mask(seg_img,torch.tensor(boxes) )

    seg_img = seg_img_t.numpy()
    seg_img = seg_img > seg_threadhold
    return boxes, classes, scores, seg_img


def merge_seg(image, seg_img, classes):
    color = Colors()
    for i in range(len(seg_img)):
        seg = seg_img[i]
        seg = seg.astype(np.uint8)
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
        seg = seg * color(classes[i])
        seg = seg.astype(np.uint8)
        image = cv2.add(image, seg)
    return image


class InferNode(Node):
    def __init__(self):
        super().__init__('yolo_infer_node')
        self.logger = self.get_logger()


        self.declare_parameter("rknn_model_file", "/home/firefly/rknn-toolkit2/rknn_toolkit_lite2/examples/yolov8_seg/yolov8_seg.rknn", ParameterDescriptor(
            name="rknn_model_file", description="Path of rknn model"))
        self.declare_parameter("rgb_image_topic", "/femto_mega/color/image_raw", ParameterDescriptor(
            name="rgb_image_topic", description="输入rgb话题，默认：/femto_mega"))
        self.declare_parameter("depth_image_topic", "/femto_mega/depth/image_raw", ParameterDescriptor(
            name="depth_image_topic", description="输入depth话题，默认：/femto_mega"))
        self.declare_parameter("camera_qos", "best_effort", ParameterDescriptor(
            name="camera_qos", description="camera using qos, best_effort or reliable"))
        self.declare_parameter("target_link", "odom", ParameterDescriptor(
            name="target_link", description="odom or map, default: odom"))

        self.rknn_model_file = self.get_parameter('rknn_model_file').value
        self.rgb_image_topic = self.get_parameter('rgb_image_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        camera_qos = self.get_parameter('camera_qos').value
        self.target_link = self.get_parameter('target_link').value


        self.qos = QoSProfile(depth=5)
        if camera_qos == 'best_effort':
            self.qos.reliability = ReliabilityPolicy.BEST_EFFORT
        elif camera_qos == 'reliable':
            self.qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            self.logger.error('Invalid value for camera_qos parameter')
            return       
        self.result_img_pub = self.create_publisher(Image, "/femto_mega/yolo_seg", self.qos)
        self.transform_stamped = TransformStamped()
        cache_time = Duration(seconds=2.0) 
        self.tf_buffer = tf2_ros.Buffer(cache_time)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        image_sub = message_filters.Subscriber(self, Image, self.rgb_image_topic, qos_profile = self.qos)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_image_topic, qos_profile = self.qos)
        time_synchronizer = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.1)
        time_synchronizer.registerCallback(self.camera_callback)
        self.marker_publisher = self.create_publisher(MarkerArray, '/femto_mega/yolo_res', 10)


        self.co_helper = COCO_test_helper(enable_letter_box=True)
        self.rknn_lite = RKNNLite()
        # Load RKNN model
        self.get_logger().info('--> Load RKNN model')
        ret = self.rknn_lite.load_rknn(self.rknn_model_file)
        if ret != 0:
            self.get_logger().error('Load RKNN model failed')
            exit(ret)
        # Init runtime environment
        self.get_logger().info('--> Init runtime environment')
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            self.get_logger().error('Init runtime environment failed')
            exit(ret)


        # cal depth
        self.fx = 750.7529296875
        self.fy = 750.503662109375
        self.cx = 636.1572875976562
        self.cy = 339.2265319824219


    def __del__(self):
        self.rknn_lite.release()


    def draw(self, image, depth, boxes, scores, classes):
        points= { }
        cnt = 0
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            # 计算中心点坐标

            center_x = (top + right) // 2
            center_y = (left + bottom) // 2
            # print(top, left, right, bottom, center_x, center_y)
            
            try:
                # Calculate the 3D coordinates
                depth_value = depth[center_x][center_y]
                if depth_value == 0:
                    continue
            except :
                continue
            cnt += 1
            x = (center_x - self.cx) * depth_value / self.fx
            y = (center_y - self.cy) * depth_value / self.fy
            z = depth_value
            cv2.circle(image, (int(center_x), int(center_y)), int(score*10), (255, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, f'{CLASSES[cl]} {score:.2f} | Position: {(x/1000):.2f} {(y/1000):.2f} {(z/1000):.2f}',
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            points[f'{CLASSES[cl]}_{cnt}'] = (x/1000, y/1000, z/1000)
        return points


    def quaternion_to_matrix(self, q):
        """Convert a quaternion into a rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
        ])

    def transform_point(self, point, transform_stamped):
        """Transform a point from frame A to frame B using TransformStamped."""
        translation = [
            transform_stamped.transform.translation.x,
            transform_stamped.transform.translation.y,
            transform_stamped.transform.translation.z
        ]
        rotation = [
            transform_stamped.transform.rotation.w,
            transform_stamped.transform.rotation.x,
            transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z
        ]
        R = self.quaternion_to_matrix(rotation)
        point = np.array([point[0], point[1], point[2]])
        transformed_point = R @ point + np.array(translation)
        return transformed_point

    def prepare_marker_array(self, points, classes, transform_stamped):
        marker_array = MarkerArray()
        for name, point in points.items():
            transformed_point = self.transform_point(point, transform_stamped)
            
            cls_name = name.split('_')[0]
            # Create and publish PointStamped message
            marker = Marker()
            marker.ns = cls_name
            marker.id = int(name.split('_')[1])
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.header = transform_stamped.header
            marker.pose.position.x = transformed_point[0]
            marker.pose.position.y = transformed_point[1]
            marker.pose.position.z = transformed_point[2]
            color = Colors()
            r,g,b = color(CLASSES.index(cls_name))
            marker.color.a = 0.6
            marker.color.r = float(r/255)
            marker.color.g = float(g/255)
            marker.color.b = float(b/255)
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.text = name

            marker_array.markers.append(marker)
        return marker_array

    def camera_callback(self, rgb_msg, depth_msg):
        try:
            transform_stamped = self.tf_buffer.lookup_transform(self.target_link, depth_msg.header.frame_id,  depth_msg.header.stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().debug(f'Failed to lookup transform {self.target_link} to {depth_msg.header.frame_id}, {ex}')
            return

        image_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")        
        image_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        img = self.co_helper.letter_box(im= image_rgb.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(114, 114, 114))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        outputs = self.rknn_lite.inference(inputs=[img])
        boxes, classes, scores, seg_img = post_process(outputs)

        if boxes is not None:
            real_boxs = self.co_helper.get_real_box(boxes)
            real_segs = self.co_helper.get_real_seg(seg_img)
            img_p = merge_seg(image_rgb, real_segs, classes)

            points = self.draw(img_p, image_depth, real_boxs, scores, classes)        
            marker_array = self.prepare_marker_array(points, classes, transform_stamped)

            result_img_msg = self.bridge.cv2_to_imgmsg(img_p, encoding="bgr8",header=rgb_msg.header) 
            self.result_img_pub.publish(result_img_msg)
            self.marker_publisher.publish(marker_array)
        return 

def main(args=None):
    rclpy.init(args=args)
    infer_node = InferNode()

    rclpy.spin(infer_node)
    infer_node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()


