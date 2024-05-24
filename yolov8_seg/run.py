import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import ReliabilityPolicy, QoSProfile
import tf2_ros
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from rcl_interfaces.msg import ParameterDescriptor



import cv2
from ultralytics import YOLO, utils
import numpy as np



# Load a model

class Annotator(utils.plotting.Annotator):
    def __init__(self, image, line_width=1):
        super().__init__(image, line_width)
    
    def custom_method(self):
        # Your custom implementation here
        pass

    def seg_bbox(self, mask, mask_color=(255, 0, 255), det_label=None, track_label=None, map_coordinate=None, img_coordinate=None, probility=None, is_valid=True):
        """
        Function for drawing segmented object in bounding box shape.

        Args:
            mask (list): masks data list for instance segmentation area plotting
            mask_color (tuple): mask foreground color
            det_label (str): Detection label text
            track_label (str): Tracking label text
        """
        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)

        label = f"Track ID: {track_label}" if track_label else ""
        label += f" | Detection: {det_label}" if det_label else ""

        text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)

        cv2.rectangle(
            self.im,
            (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
            (int(mask[0][0]) + text_size[0] // 2 + 5, int(mask[0][1] + 5)),
            mask_color,
            -1,
        )

        cv2.putText(
            self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, (255, 255, 255), 2
        )

        if is_valid:
            cv2.circle(self.im, (int(img_coordinate[0]), int(img_coordinate[1])), int(probility*10), mask_color, -1, lineType=cv2.LINE_AA)
            marker = Marker()
            marker.ns = det_label
            marker.id = int(track_label)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            # marker.header = transform_stamped.header
            marker.pose.position.x = map_coordinate[0]
            marker.pose.position.y = map_coordinate[1]
            marker.pose.position.z = map_coordinate[2]
            r,g,b = mask_color
            marker.color.a = 0.6
            marker.color.r = float(r/255)
            marker.color.g = float(g/255)
            marker.color.b = float(b/255)
            marker.scale.x = 0.1 * probility
            marker.scale.y = 0.1 * probility
            marker.scale.z = 0.1 * probility
            marker.text = label
            return marker
        else:
            return None
        


class InferNode(Node):
    def __init__(self):
        super().__init__('yolo_infer_node')
        self.logger = self.get_logger()

        self.declare_parameter("model_file", "/home/firefly/yolov9c-seg.pt", ParameterDescriptor(
            name="model_file", description="Path of model"))
        self.declare_parameter("rgb_image_topic", "/femto_mega/color/image_raw", ParameterDescriptor(
            name="rgb_image_topic", description="输入rgb话题，默认：/femto_mega"))
        self.declare_parameter("depth_image_topic", "/femto_mega/depth/image_raw", ParameterDescriptor(
            name="depth_image_topic", description="输入depth话题，默认：/femto_mega"))
        self.declare_parameter("camera_qos", "best_effort", ParameterDescriptor(
            name="camera_qos", description="camera using qos, best_effort or reliable"))

        # 获取ROS参数的值
        self.model_file = self.get_parameter('model_file').value
        self.rgb_image_topic = self.get_parameter('rgb_image_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        camera_qos = self.get_parameter('camera_qos').value

        self.qos = QoSProfile(depth=5)
        if camera_qos == 'best_effort':
            self.qos.reliability = ReliabilityPolicy.BEST_EFFORT
        elif camera_qos == 'reliable':
            self.qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            self.logger.error('Invalid value for camera_qos parameter')
            return       

        self.model = YOLO(self.model_file)  # load a pretrained model

        # cal depth
        self.fx = 750.7529296875
        self.fy = 750.503662109375
        self.cx = 636.1572875976562
        self.cy = 339.2265319824219

        self.bridge = CvBridge()
        camera_name = self.rgb_image_topic.split('/')[1]
        image_sub = message_filters.Subscriber(self, Image, self.rgb_image_topic, qos_profile = self.qos)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_image_topic, qos_profile = self.qos)
        time_synchronizer = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.1, allow_headerless=True)
        time_synchronizer.registerCallback(self.camera_callback)
        self.transform_stamped = TransformStamped()
        cache_time = Duration(seconds=2.0) 
        self.tf_buffer = tf2_ros.Buffer(cache_time)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.result_img_pub = self.create_publisher(Image, f"/{camera_name}/yolo_seg", self.qos)
        self.marker_publisher = self.create_publisher(MarkerArray, f'/{camera_name}/yolo_markers', 10)


    def calculate_3d_coordinates(self, u, v, depth_value):
        """Calculate the 3D coordinates"""
        x = (u - self.cx) * depth_value / self.fx
        y = (v - self.cy) * depth_value / self.fy
        z = depth_value

        return x / 1000, y / 1000, z / 1000


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


    def camera_callback(self, rgb_msg, depth_msg):
        try:
            transform_stamped = self.tf_buffer.lookup_transform('odom', depth_msg.header.frame_id,  depth_msg.header.stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().debug(f'Failed to lookup transform odom to {depth_msg.header.frame_id}, {ex}')
            return

        image_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")        
        image_pixel = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        annotator = Annotator(image_rgb, line_width=2)
        results = self.model.track(image_rgb, show=False, vid_stride=2, device='cuda:0', iou=0.9) 

        if results[0].boxes.id is not None and results[0].masks is not None:
            marker_array = MarkerArray()

            cls_ids = results[0].boxes.cls.int().cpu().tolist()
            conf = results[0].boxes.conf.cpu().tolist()
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            names = results[0].names
            for mask, track_id, cls_id, prob in zip(masks, track_ids, cls_ids, conf):
                # 初始化边界变量
                min_x = float('inf')
                max_x = float('-inf')
                min_y = float('inf')
                max_y = float('-inf')
                for point in mask:
                    x, y = point
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                # 计算中心点坐标
                center_x = int((min_x + max_x) / 2)
                center_y = int((min_y + max_y) / 2)

                map_coordinate_cam_link = self.calculate_3d_coordinates(center_x, center_y, image_pixel[center_y, center_x])
                is_valid = True
                if map_coordinate_cam_link[0] == 0 and map_coordinate_cam_link[1] == 0 and map_coordinate_cam_link[2] == 0:
                    is_valid = False
                map_coordinate_map_link = self.transform_point(map_coordinate_cam_link, transform_stamped)

                marker = annotator.seg_bbox(mask=mask, mask_color=utils.plotting.colors(track_id, True), track_label=str(track_id), det_label=names[cls_id], map_coordinate=map_coordinate_map_link, img_coordinate=(center_x, center_y), probility=prob, is_valid=is_valid)
                if is_valid:
                    marker.header = transform_stamped.header
                    # print(map_coordinate_cam_link,map_coordinate_map_link,marker)
                    marker_array.markers.append(marker)

            self.marker_publisher.publish(marker_array)

        # # 创建窗口并显示拼接后的图像
        # cv2.imshow("RGB", image_rgb)
        # key = cv2.waitKey(1)
        result_img_msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding="bgr8",header=rgb_msg.header) 
        self.result_img_pub.publish(result_img_msg)



        return 
    

def main(args=None):
    rclpy.init(args=args)

    infer_node = InferNode()

    try:
        rclpy.spin(infer_node)
    except KeyboardInterrupt:
        pass

    infer_node.destroy_node()
    rclpy.shutdown()