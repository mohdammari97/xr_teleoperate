import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera(object):
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False) -> None:
        """
        img_shape: [height, width]
        serial_number: serial number
        """
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.init_realsense()

    def init_realsense(self):

        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number is not None:
            config.enable_device(self.serial_number)

        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self._device is None:
            print('[Image Server] pipe_profile.get_device() is None .')
        if self.enable_depth:
            assert self._device is not None
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = np.asanyarray(depth_frame.get_data()) if self.enable_depth else None
        return color_image, depth_image

    def release(self):
        self.pipeline.stop()


class OpenCVCamera():
    def __init__(self, device_id, img_shape, fps):
        """
        decive_id: /dev/video* or *
        img_shape: [height, width]
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Test if the camera can read frames
        if not self._can_read_frame():
            print(f"[Image Server] Camera {self.id} Error: Failed to initialize the camera or read frames. Exiting...")
            self.release()

    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def release(self):
        self.cap.release()

    def get_frame(self):
        ret, color_image = self.cap.read()
        if not ret:
            return None
        return color_image


class ImageServer:
    def __init__(self, config, port = 5555, Unit_Test = False):
        """
        config example1:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            'wrist_camera_type': 'realsense', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # realsense camera's serial number
        }

        config example2:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'realsense',                                  # opencv or realsense
            'head_camera_image_shape': [480, 640],                            # Head camera resolution  [height, width]
            'head_camera_id_numbers': ["218622271739"],                       # realsense camera's serial number
            'wrist_camera_type': 'opencv', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': [0,1],                                 # '/dev/video0' and '/dev/video1' (opencv)
        }

        If you are not using the wrist camera, you can comment out its configuration, like this below:
        config:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            #'wrist_camera_type': 'realsense', 
            #'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            #'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # serial number (realsense)
        }
        """
        print(config)
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])      # (height, width)
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])
        
        # Active camera configuration (separate from head camera)
        self.active_camera_type = config.get('active_camera_type', None)
        self.active_image_shape = config.get('active_camera_image_shape', [720, 1280]) #(height, width)
        self.active_camera_id_numbers = config.get('active_camera_id_numbers', None)
        

        self.wrist_camera_type = config.get('wrist_camera_type', None)
        self.wrist_image_shape = config.get('wrist_camera_image_shape', [480, 640])    # (height, width)
        self.wrist_camera_id_numbers = config.get('wrist_camera_id_numbers', None)

        self.port = port
        self.Unit_Test = Unit_Test


        # Initialize head cameras
        self.head_cameras = []
        if self.head_camera_type == 'opencv':
            for device_id in self.head_camera_id_numbers:
                camera = OpenCVCamera(device_id=device_id, img_shape=self.head_image_shape, fps=self.fps)
                self.head_cameras.append(camera)
        elif self.head_camera_type == 'realsense':
            for serial_number in self.head_camera_id_numbers:
                camera = RealSenseCamera(img_shape=self.head_image_shape, fps=self.fps, serial_number=serial_number)
                self.head_cameras.append(camera)
        else:
            print(f"[Image Server] Unsupported head_camera_type: {self.head_camera_type}")

        # Initialize active cameras if configured
        self.active_cameras = []
        if self.active_camera_type == 'opencv':
            for device_id in self.active_camera_id_numbers:
                camera = OpenCVCamera(device_id=device_id, img_shape=self.active_image_shape, fps=self.fps)
                self.active_cameras.append(camera)
        elif self.active_camera_type == 'realsense':
            for serial_number in self.active_camera_id_numbers:
                camera = RealSenseCamera(img_shape=self.active_image_shape, fps=self.fps, serial_number=serial_number)
                self.active_cameras.append(camera)
        else:
            print(f"[Image Server] Unsupported active_camera_type: {self.active_camera_type}")

        # Initialize wrist cameras if provided
        self.wrist_cameras = []
        if self.wrist_camera_type and self.wrist_camera_id_numbers:
            if self.wrist_camera_type == 'opencv':
                for i, device_id in enumerate(self.wrist_camera_id_numbers):
                    # Use standard wrist shape for all cameras
                    camera = OpenCVCamera(device_id=device_id, img_shape=self.wrist_image_shape, fps=self.fps)
                    self.wrist_cameras.append(camera)
            elif self.wrist_camera_type == 'realsense':
                for serial_number in self.wrist_camera_id_numbers:
                    camera = RealSenseCamera(img_shape=self.wrist_image_shape, fps=self.fps, serial_number=serial_number)
                    self.wrist_cameras.append(camera)
            else:
                print(f"[Image Server] Unsupported wrist_camera_type: {self.wrist_camera_type}")

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        
        # Create second socket for active camera full resolution stream if needed
        self.active_socket = self.context.socket(zmq.PUB)
        self.active_socket.bind(f"tcp://*:{self.port + 1}")  # Use port+1 for active camera
        print(f"[Image Server] Active camera full resolution stream on port {self.port + 1}")
        print(f"[Image Server] Head/wrist concatenated stream on port {self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        for cam in self.head_cameras:
            if isinstance(cam, OpenCVCamera):
                print(f"[Image Server] Head camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                print(f"[Image Server] Head camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                print("[Image Server] Unknown camera type in head_cameras.")

        for cam in self.active_cameras:
            if isinstance(cam, OpenCVCamera):
                print(f"[Image Server] Active camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                print(f"[Image Server] Active camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                print("[Image Server] Unknown camera type in active cameras.")

        for i, cam in enumerate(self.wrist_cameras):
            if isinstance(cam, OpenCVCamera):
                print(f"[Image Server] Wrist camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                print(f"[Image Server] Wrist camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                print("[Image Server] Unknown camera type in wrist_cameras.")

        print("[Image Server] Image server has started, waiting for client connections...")



    def _init_performance_metrics(self):
        self.frame_count = 0  # Total frames sent
        self.time_window = 1.0  # Time window for FPS calculation (in seconds)
        self.frame_times = deque()  # Timestamps of frames sent within the time window
        self.start_time = time.time()  # Start time of the streaming

    def _update_performance_metrics(self, current_time):
        # Add current time to frame times deque
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        # Increment frame count
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            print(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def _close(self):
        for cam in self.head_cameras:
            cam.release()
        for cam in self.active_cameras:
            cam.release()
        for cam in self.wrist_cameras:
            cam.release()
        self.socket.close()
        self.active_socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    def send_process(self):
        try:
            while True:
                if self.active_cameras:
                    # Active camera mode: send full resolution active camera + head/wrist stream
                    self._send_active_camera_streams()
                else:
                    # Head camera only mode: send head/wrist concatenated stream
                    self._send_head_camera_stream()

                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            self._close()

    def _send_active_camera_streams(self):
        """Send active camera at full resolution + head/wrist concatenated stream."""
        # Get active camera frame at full resolution
        active_frames = []
        for cam in self.active_cameras:  # Use actual active cameras
            if self.active_camera_type == 'opencv':
                color_image = cam.get_frame()
                if color_image is None:
                    print("[Image Server] Active camera frame read is error.")
                    return
                # Rotate the image by 180 degrees for active camera
                color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                active_frames.append(color_image)
            elif self.active_camera_type == 'realsense':
                color_image, depth_image = cam.get_frame()
                if color_image is None:
                    print("[Image Server] Active camera frame read is error.")
                    return
                active_frames.append(color_image)
        
        if len(active_frames) != len(self.active_cameras):
            return
        
        # Concatenate active camera frames and send at full resolution
        active_full_res = cv2.hconcat(active_frames)
        
        # Send full resolution active camera stream on port+1
        ret, buffer = cv2.imencode('.jpg', active_full_res)
        if ret:
            jpg_bytes = buffer.tobytes()
            if self.Unit_Test:
                timestamp = time.time()
                frame_id = self.frame_count
                header = struct.pack('dI', timestamp, frame_id)
                message = header + jpg_bytes
            else:
                message = jpg_bytes
            self.active_socket.send(message)
        
        # Now send the concatenated head/wrist stream for recording
        self._send_head_camera_stream()

    def _send_head_camera_stream(self):
        """Send head camera cropped + wrist concatenated stream."""
        head_frames = []
        for cam in self.head_cameras:
            if self.head_camera_type == 'opencv':
                color_image = cam.get_frame()
                if color_image is None:
                    print("[Image Server] Head camera frame read is error.")
                    return
                # Archive: Crop to center of head camera
                # --------------------------------------
                # # Head camera: crop to 480×1280 region
                # h, w = color_image.shape[:2]          # h=1080, w=3840
                # half_w = w // 2                        # 1920 pixels per eye

                # # crop height
                # new_h, new_w = 480, 640
                # start_y = (h - new_h) // 2            # (1080-480)//2 = 300
                # start_x_local = (half_w - new_w) // 2  # (1920-640)//2 = 640
                
                # # crop center of left eye
                # left_crop = color_image[
                #     start_y:start_y+new_h,
                #     start_x_local:start_x_local+new_w
                # ]

                # # crop center of right eye
                # right_crop = color_image[
                #     start_y:start_y+new_h,
                #     half_w + start_x_local : half_w + start_x_local + new_w
                # ]

                # # stitch them back side by side
                # color_image = cv2.hconcat([left_crop, right_crop])
                # ---------------------------------------
                
                # Split stereo image and crop edges from both halves
                h, w = color_image.shape[:2]  # h=1080, w=3840
                half_w = w // 2  # 1920 pixels per eye
                
                # Crop parameters for each eye
                crop_w_ratio = 0.6  # Keep 80% of width (remove 10% from each side)
                crop_h_ratio = 0.7  # Keep 80% of height (remove from bottom only)
                
                new_eye_w = int(half_w * crop_w_ratio)  # 1920 * 0.8 = 1536
                new_h = int(h * crop_h_ratio)  # 1080 * 0.8 = 864
                
                start_x_local = (half_w - new_eye_w) // 2  # Center crop within each eye
                start_y = 0  # Start from top (no cropping from top)
                
                # Crop left eye (first half) - keep top, crop bottom and sides
                left_eye = color_image[
                    start_y:start_y+new_h,
                    start_x_local:start_x_local+new_eye_w
                ]
                
                # Crop right eye (second half) - keep top, crop bottom and sides
                right_eye = color_image[
                    start_y:start_y+new_h,
                    half_w + start_x_local:half_w + start_x_local + new_eye_w
                ]
                
                # Stitch the cropped eyes back together
                color_image = cv2.hconcat([left_eye, right_eye])
                
                # Downsample the stitched image to target resolution
                color_image = cv2.resize(color_image, (1280, 480))

                # Rotate the image by 180 degrees for head camera
                color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                
            elif self.head_camera_type == 'realsense':
                color_image, depth_image = cam.get_frame()
                if color_image is None:
                    print("[Image Server] Head camera frame read is error.")
                    return
            
            head_frames.append(color_image)
        
        if len(head_frames) != len(self.head_cameras):
            return
        
        head_color = cv2.hconcat(head_frames)
        
        # Handle wrist cameras
        if self.wrist_cameras:
            wrist_frames = []
            target_height, target_width = self.wrist_image_shape  # 480, 640
            
            for i, cam in enumerate(self.wrist_cameras):
                if self.wrist_camera_type == 'opencv':
                    color_image = cam.get_frame()
                    if color_image is None:
                        print("[Image Server] Wrist camera frame read is error.")
                        return
                    
                    # Apply different rotations based on camera index
                    if i == 0:  # Left wrist camera
                        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                    elif i == 1:  # Right wrist camera
                        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                    
                    # Ensure all cameras have the same final dimensions
                    # Resize to target dimensions regardless of input size
                    color_image = cv2.resize(color_image, (target_width, target_height))
                        
                elif self.wrist_camera_type == 'realsense':
                    color_image, depth_image = cam.get_frame()
                    if color_image is None:
                        print("[Image Server] Wrist camera frame read is error.")
                        return
                
                wrist_frames.append(color_image)
            
            if len(wrist_frames) == len(self.wrist_cameras):
                wrist_color = cv2.hconcat(wrist_frames)
                # Concatenate head and wrist frames
                full_color = cv2.hconcat([head_color, wrist_color])
            else:
                full_color = head_color
        else:
            full_color = head_color

        # Send concatenated stream
        ret, buffer = cv2.imencode('.jpg', full_color)
        if ret:
            jpg_bytes = buffer.tobytes()
            if self.Unit_Test:
                timestamp = time.time()
                frame_id = self.frame_count
                header = struct.pack('dI', timestamp, frame_id)
                message = header + jpg_bytes
            else:
                message = jpg_bytes
            self.socket.send(message)


if __name__ == "__main__":
    config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [1080, 3840], #,[480, 1280], # Head camera resolution
        'head_camera_id_numbers': [0],
        'active_camera_type': 'opencv',
        'active_camera_image_shape': [720, 2560], # Resolution of active cam
        'active_camera_id_numbers': [4],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
        'wrist_camera_id_numbers': [2, 6],
    }

    server = ImageServer(config, Unit_Test=False)
    server.send_process()
    