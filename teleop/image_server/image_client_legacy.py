import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory

class ImageClient:
    def __init__(self, vr_img_shape = None, vr_img_shm_name = None, wrist_img_shape = None, wrist_img_shm_name = None,
                 head_cam_img_shape = None, head_cam_img_shm_name = None, use_active_camera = True,
                 active_cam_img_shape = None, active_cam_img_shm_name = None,
                 image_show = False, server_address = "192.168.123.164", port = 5555, Unit_Test = False):
        """
        vr_img_shape: User's expected VR display resolution shape (H, W, C). Can be full active camera or cropped head camera.

        vr_img_shm_name: Shared memory for VR display images.
        
        head_cam_img_shape: Head camera resolution shape (H, W, C). Always 480x1280 for dataset compatibility.
        
        head_cam_img_shm_name: Shared memory for head cam images.

        active_cam_img_shape: Active head camera resolution shape (H, W, C). Always 480x1280 for dataset compatibility.
        
        active_cam_img_shm_name: Shared memory for active head cam images.

        wrist_img_shape: User's expected wrist camera resolution shape (H, W, C).

        wrist_img_shm_name: Shared memory is used to easily transfer images.
        
        use_active_camera: Whether to use active camera (True) or head camera (False).
        
        image_show: Whether to display received images in real time.

        server_address: The ip address to execute the image server script.

        port: The port number to bind to. It should be the same as the image server.

        Unit_Test: When both server and client are True, it can be used to test the image transfer latency, \
                   network jitter, frame loss rate and other information.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port
        self.use_active_camera = use_active_camera

        self.vr_img_shape = vr_img_shape
        self.head_cam_img_shape = head_cam_img_shape
        self.wrist_img_shape = wrist_img_shape
        self.active_cam_img_shape = active_cam_img_shape

        # Set up TV/VR display shared memory
        self.vr_enable_shm = False
        if self.vr_img_shape is not None and vr_img_shm_name is not None:
            self.vr_image_shm = shared_memory.SharedMemory(name=vr_img_shm_name)
            self.vr_img_array = np.ndarray(vr_img_shape, dtype = np.uint8, buffer = self.vr_image_shm.buf)
            self.vr_enable_shm = True
        
        # Set up head_cam shared memory
        self.head_cam_enable_shm = False
        if self.head_cam_img_shape is not None and head_cam_img_shm_name is not None:
            self.head_camimage_shm = shared_memory.SharedMemory(name=head_cam_img_shm_name)
            self.head_cam_img_array = np.ndarray(head_cam_img_shape, dtype = np.uint8, buffer = self.head_camimage_shm.buf)
            self.head_cam_enable_shm = True
        
        # Set up wrist shared memory
        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        # Set up active camera shared memory (separate from recording)
        self.active_cam_enable_shm = False
        if self.active_cam_img_shape is not None and active_cam_img_shm_name is not None:
            self.active_cam_image_shm = shared_memory.SharedMemory(name=active_cam_img_shm_name)
            self.active_cam_img_array = np.ndarray(active_cam_img_shape, dtype = np.uint8, buffer = self.active_cam_image_shm.buf)
            self.active_cam_enable_shm = True

        # Performance evaluation parameters
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID

        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window

        # Data transmission quality metrics
        self._latencies = deque()  # Latencies of frames within the time window
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # Update latency
        latency = receive_time - timestamp
        self._latencies.append(latency)

        # Remove latencies outside the time window
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        # Update frame counts for lost frame calculation
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                print(f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}")
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1

        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # Calculate real-time FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # Calculate latency metrics
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            print(f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, \
                  Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%")
    
    def _close(self):
        self._socket.close()
        if hasattr(self, '_active_socket'):
            self._active_socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        print("Image client has been closed.")

    
    def receive_process(self):
        """Main receive process that handles both single and dual stream modes."""
        if self.use_active_camera:
            self._receive_dual_streams()
        else:
            self._receive_single_stream()

    def _receive_single_stream(self):
        """Receive head camera + wrist concatenated stream."""
        # Set up ZeroMQ context and socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print("\n[Image Client] Single stream mode - waiting for head/wrist data...")
        try:
            while self.running:
                # Receive message
                message = self._socket.recv()
                receive_time = time.time()

                if self._enable_performance_eval:
                    header_size = struct.calcsize('dI')
                    try:
                        # Attempt to extract header and image data
                        header = message[:header_size]
                        jpg_bytes = message[header_size:]
                        timestamp, frame_id = struct.unpack('dI', header)
                    except struct.error as e:
                        print(f"[Image Client] Error unpacking header: {e}, discarding message.")
                        continue
                else:
                    # No header, entire message is image data
                    jpg_bytes = message
                # Decode image
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if current_image is None:
                    print("[Image Client] Failed to decode image.")
                    continue

                # Process head camera image (already cropped on server side to 480x1280)
                height, width = current_image.shape[:2]
                
                # Extract head/wrist parts
                if self.wrist_enable_shm and width > 1280: # TODO Was genau passiert hier?
                    # Image contains both head (480x1280) and wrist cameras
                    head_image = current_image[:, :1280]  # First 1280 pixels are head
                    wrist_image = current_image[:, 1280:]  # Remaining pixels are wrist
                    np.copyto(self.wrist_img_array, wrist_image)
                else:
                    # Only head camera
                    head_image = current_image

                # Copy to VR shm for display
                if self.vr_enable_shm:
                    if head_image.shape[:2] != (self.vr_img_shape[0], self.vr_img_shape[1]):
                        head_image = cv2.resize(head_image, (self.vr_img_shape[1], self.vr_img_shape[0]))
                    np.copyto(self.vr_img_array, head_image)

                # Copy to head cam shm for recording
                if self.head_cam_enable_shm:
                    if head_image.shape[:2] != (self.head_cam_img_shape[0], self.head_cam_img_shape[1]):
                        head_image = cv2.resize(head_image, (self.head_cam_img_shape[1], self.head_cam_img_shape[0]))
                    np.copyto(self.head_cam_img_array, head_image)

                if self._image_show:
                    height, width = current_image.shape[:2]
                    resized_image = cv2.resize(current_image, (width // 2, height // 2))
                    cv2.imshow('Image Client Stream', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                if self._enable_performance_eval:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                    self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()

    def _receive_dual_streams(self):
        """Receive both full resolution active camera stream and concatenated stream."""
        # Set up ZeroMQ context and sockets
        self._context = zmq.Context()
        
        # Socket for concatenated stream (port)
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Socket for full resolution active camera stream (port+1)
        self._active_socket = self._context.socket(zmq.SUB)
        self._active_socket.connect(f"tcp://{self._server_address}:{self._port + 1}")
        self._active_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Set up poller for non-blocking receive
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        poller.register(self._active_socket, zmq.POLLIN)

        print(f"\n[Image Client] Dual stream mode - waiting for active camera (port {self._port + 1}) and concatenated data (port {self._port})...")
        
        # Add small delay to allow connections to establish
        time.sleep(1.0)

        try:
            while self.running:
                # Poll for messages with longer timeout
                socks = dict(poller.poll(timeout=1000))  # 1 second timeout
                
                if not socks:
                    print("[Image Client] No data received in 1 second, continuing...")
                    continue
                
                # Handle full resolution active camera stream (for VR display)
                if self._active_socket in socks:

                    message = self._active_socket.recv(zmq.NOBLOCK)
                    receive_time = time.time()

                    if self._enable_performance_eval:
                        header_size = struct.calcsize('dI')
                        try:
                            header = message[:header_size]
                            jpg_bytes = message[header_size:]
                            timestamp, frame_id = struct.unpack('dI', header)
                        except struct.error as e:
                            print(f"[Image Client] Error unpacking active camera header: {e}")
                            continue
                    else:
                        jpg_bytes = message

                    # Decode full resolution active camera image
                    np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    active_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    if active_image is not None:
                        # Store for VR display at full resolution
                        if self.vr_enable_shm:
                            # Use full resolution for VR display
                            if active_image.shape[:2] != (self.vr_img_shape[0], self.vr_img_shape[1]):
                                processed_active_image = cv2.resize(active_image, (self.vr_img_shape[1], self.vr_img_shape[0]))
                            else:
                                processed_active_image = active_image
                            np.copyto(self.vr_img_array, processed_active_image)
                        
                        # Downscale active camera for recording and put in separate active camera shared memory
                        if self.active_cam_enable_shm:
                            # Downscale from 720x2560 to 480x1280 for recording
                            active_downscaled = cv2.resize(active_image, (1280, 480))
                            np.copyto(self.active_cam_img_array, active_downscaled)
                        
                        # Display active camera in standalone mode
                        if self._image_show:
                            height, width = active_image.shape[:2]
                            resized_active = cv2.resize(active_image, (width // 2, height // 2))
                            cv2.imshow('Image Client Active Camera', resized_active)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.running = False

                # Handle concatenated stream (for recording and wrist cameras)
                if self._socket in socks:
                    
                    message = self._socket.recv(zmq.NOBLOCK)
                    receive_time = time.time()

                    if self._enable_performance_eval:
                        header_size = struct.calcsize('dI')
                        try:
                            header = message[:header_size]
                            jpg_bytes = message[header_size:]
                            timestamp, frame_id = struct.unpack('dI', header)
                        except struct.error as e:
                            print(f"[Image Client] Error unpacking concatenated header: {e}")
                            continue
                    else:
                        jpg_bytes = message

                    # Decode concatenated image
                    np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    concat_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    if concat_image is None:
                        continue

                    height, width = concat_image.shape[:2]
                    

                    # Concatenated stream contains head + wrist
                    if self.wrist_enable_shm and width > 1280:
                        # Image contains both head (480x1280) and wrist cameras
                        head_image = concat_image[:, :1280]  # First 1280 pixels are head
                        wrist_image = concat_image[:, 1280:]  # Remaining pixels are wrist
                        np.copyto(self.wrist_img_array, wrist_image)
                        
                        # Put head camera in recording shared memory
                        if self.head_cam_enable_shm:
                            if head_image.shape[:2] != (self.head_cam_img_shape[0], self.head_cam_img_shape[1]):
                                head_image = cv2.resize(head_image, (self.head_cam_img_shape[1], self.head_cam_img_shape[0]))
                            np.copyto(self.head_cam_img_array, head_image)
                    else:
                        # Only head camera
                        head_image = concat_image
                        if self.head_cam_enable_shm:
                            if head_image.shape[:2] != (self.head_cam_shape[0], self.head_cam_img_shape[1]):
                                head_image = cv2.resize(head_image, (self.head_cam_img_shape[1], self.head_cam_img_shape[0]))
                            np.copyto(self.head_cam_img_array, head_image)

                    if self._image_show:
                        height, width = concat_image.shape[:2]
                        resized_image = cv2.resize(concat_image, (width // 2, height // 2))
                        cv2.imshow('Image Client Concatenated Stream', resized_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.running = False

                    if self._enable_performance_eval:
                        self._update_performance_metrics(timestamp, frame_id, receive_time)
                        self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving dual streams: {e}")
        finally:
            self._close()

if __name__ == "__main__":
    # example1
    # tv_img_shape = (480, 1280, 3)
    # img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    # img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    # img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    # img_client.receive_process()

    # example2
    # Initialize the client with performance evaluation enabled
    # client = ImageClient(image_show = True, server_address='127.0.0.1', Unit_Test=True) # local test
    client = ImageClient(image_show = True, server_address='192.168.123.164', Unit_Test=False, use_active_camera=True) # deployment test with active camera
    client.receive_process()