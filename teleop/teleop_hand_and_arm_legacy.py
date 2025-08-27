import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading
import logging
import os

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.robot_control.active_head_cam import ActiveCameraController
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.pose_logger import PoseLogger

# Configure logging
def setup_logging(verbose=False):
    """Set up logging for the application"""
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure basic logging
    log_level = logging.INFO if not verbose else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "teleop_hand_and_arm.log")),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    # Set specific loggers to different levels
    # Keep the TV wrapper quiet unless in verbose mode
    logging.getLogger('tv_wrapper').setLevel(logging.WARNING if not verbose else logging.DEBUG)
    
    # Only show warnings and errors on console by default
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Replace the console handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger('teleop_hand_and_arm')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = int, default = 30.0, help = 'save data\'s frequency')

    parser.add_argument('--record', action = 'store_true', help = 'Save data or not')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = 'Do not save data')
    parser.set_defaults(record = False)

    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--hand', type=str, choices=['dex3', 'gripper', 'inspire1'], help='Select hand controller')
    parser.add_argument('--retargeting-method', type=str, choices=['vector', 'dexpilot'], default='dexpilot', 
                      help='Select hand retargeting method: vector (default) or dexpilot')

    parser.add_argument('--cyclonedds_uri', type=str, default='enxa0cec8616f27', help='Network interface for CycloneDX (default: enxa0cec8616f27)')
    # Speed Limit
    parser.add_argument('--arm-speed', type=float, default=None, 
                      help='Set the arm velocity limit (default is controller-specific)')
    parser.add_argument('--no-gradual-speed', action='store_true',
                      help='Disable gradual speed increase')
    
    # Active Camera options
    parser.add_argument('--active-camera', action='store_true', default=True, help='Enable active camera head tracking')
    parser.add_argument('--camera-port', type=str, default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3R4A5A-if00-port0", 
                       help='Serial port for the active camera servo controller')
    parser.add_argument('--camera-safe-mode', action='store_true', default=False, help='Enable safe mode with limited camera movement')
    parser.add_argument('--camera-max-movement', type=float, default=60.0, help='Maximum camera movement in degrees from start position (default: 1.0° for safety)')
    
    # Logging options
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-pose-logging', dest='pose_logging', action='store_false', help='Disable background pose logging')
    parser.set_defaults(pose_logging=True)
    parser.add_argument('--force', action='store_true', help='If set, record real qvel/torque for arms and hands (default: False)')
    parser.set_defaults(force=False)

    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info(f"Starting teleop_hand_and_arm.py with args: {args}")

    # Initialize pose logger if enabled
    pose_logger = None
    if args.pose_logging:
        pose_logger = PoseLogger(
            log_dir=os.path.join(current_dir, "logs", "pose_data"),
            log_interval=1.0,  # Save to disk every 1 second
            max_buffer_size=100  # Or after 100 frames, whichever comes first
        ).start()
        logger.info(f"Background pose logging enabled, saving to {pose_logger.log_file}")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [1080, 3840], #[480, 1280],# [1080, 3840], #[480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [6],
        'active_camera_type': 'opencv',
        'active_camera_image_shape': [720, 2560], # Resolution of active cam
        'active_camera_id_numbers': [12],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
        'wrist_camera_id_numbers': [8, 10],
    }
    # Configure VR display resolutions based on active camera selection
    if args.active_camera:
        # Use active camera for VR headset with full resolution
        vr_img_shape = (img_config['active_camera_image_shape'][0], img_config['active_camera_image_shape'][1], 3)  # 720x2560 for VR
        logger.info("Using active camera for VR headset with full resolution (720x2560), recording at (480x1280)")
    else:
        # Use head camera
        vr_img_shape = (480, 1280, 3)  # Standard head camera resolution for VR (cropped from 1080x3840)
        logger.info("Using head camera for VR headset with cropped resolution (480x1280)")

    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (vr_img_shape[1] / vr_img_shape[0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    # Set up shared memory for VR display (full resolution or cropped)
    if BINOCULAR and not (vr_img_shape[1] / vr_img_shape[0] > ASPECT_RATIO_THRESHOLD):
        vr_img_shape = (vr_img_shape[0], vr_img_shape[1] * 2, 3)

    # 1. tv_shared memory for transmitting the robot's head camera image to the XR device.
    vr_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(vr_img_shape) * np.uint8().itemsize)
    vr_img_array = np.ndarray(vr_img_shape, dtype = np.uint8, buffer = vr_img_shm.buf)

    # 2. head_cam_shared memory for recording
    head_cam_img_shape = (480, 1280, 3)
    head_cam_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(head_cam_img_shape) * np.uint8().itemsize)
    head_cam_img_array = np.ndarray(head_cam_img_shape, dtype = np.uint8, buffer = head_cam_img_shm.buf)

    # 3. active camera shared memory for recording
    active_cam_img_shape = (480, 1280, 3)
    active_cam_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(active_cam_img_shape) * np.uint8().itemsize)
    active_cam_img_array = np.ndarray(active_cam_img_shape, dtype = np.uint8, buffer = active_cam_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        # 4. wrist camera shared memory for recording
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(
            vr_img_shape = vr_img_shape, 
            vr_img_shm_name = vr_img_shm.name,
            head_cam_img_shape = head_cam_img_shape,
            head_cam_img_shm_name = head_cam_img_shm.name,
            wrist_img_shape = wrist_img_shape, 
            wrist_img_shm_name = wrist_img_shm.name,
            active_cam_img_shape = active_cam_img_shape if args.active_camera else None,
            active_cam_img_shm_name = active_cam_img_shm.name if args.active_camera else None,
            use_active_camera = args.active_camera
        )
    else:
        img_client = ImageClient(
            vr_img_shape = vr_img_shape, 
            vr_img_shm_name = vr_img_shm.name,
            head_cam_img_shape = head_cam_img_shape,
            head_cam_img_shm_name = head_cam_img_shm.name,
            active_cam_img_shape = active_cam_img_shape if args.active_camera else None,
            active_cam_img_shm_name = active_cam_img_shm.name if args.active_camera else None,
            use_active_camera = args.active_camera
        )

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()
    logger.info("Image receive thread started")

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVisionWrapper(BINOCULAR, vr_img_shape, vr_img_shm.name, ngrok=True)
    logger.info("TeleVision wrapper initialized")

    # arm
    if args.arm == 'G1_29':
        arm_ctrl = G1_29_ArmController(networkInterface=args.cyclonedds_uri)
        arm_ik = G1_29_ArmIK()
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed
            logger.info(f"Setting custom arm velocity limit: {args.arm_speed}")
    elif args.arm == 'G1_23':
        arm_ctrl = G1_23_ArmController(networkInterface=args.cyclonedds_uri)
        arm_ik = G1_23_ArmIK()
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed
    elif args.arm == 'H1_2':
        arm_ctrl = H1_2_ArmController(networkInterface=args.cyclonedds_uri)
        arm_ik = H1_2_ArmIK()
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed
    elif args.arm == 'H1':
        arm_ctrl = H1_ArmController()
        arm_ik = H1_ArmIK()
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed

    # active camera
    camera_controller = None
    if args.active_camera:
        try:
            logger.info(f"Initializing active camera controller on port: {args.camera_port}")
            camera_controller = ActiveCameraController(
                port=args.camera_port,
                safe_mode=args.camera_safe_mode,
                max_movement_deg=args.camera_max_movement,
                logger=logger
            )
            if camera_controller.connect():
                logger.info("Active camera controller initialized successfully")
                # Get initial positions for reference
                initial_pitch, initial_yaw = camera_controller.get_positions()
                logger.info(f"Initial camera positions - Pitch: {np.degrees(initial_pitch):.1f}°, Yaw: {np.degrees(initial_yaw):.1f}°")
            else:
                logger.warning("Failed to connect to active camera controller")
                camera_controller = None
        except Exception as e:
            logger.error(f"Failed to initialize active camera controller: {e}")
            camera_controller = None

    # handbased on the 
    if args.hand == "dex3":
        # Dynamically set shared array size based on --force
        if args.force:
            hand_state_size = 66  # 33 for left, 33 for right
        else:
            hand_state_size = 38  # 19 for left, 19 for right
        left_hand_array = Array('d', 75, lock = True)         # [input]
        right_hand_array = Array('d', 75, lock = True)        # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', hand_state_size, lock = False)  # [output] current left, right hand state
        dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, networkInterface=args.cyclonedds_uri, force=args.force, retargeting_method=args.retargeting_method)
    elif args.hand == "gripper":
        left_hand_array = Array('d', 75, lock=True)
        right_hand_array = Array('d', 75, lock=True)
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array, networkInterface=args.cyclonedds_uri)
    elif args.hand == "inspire1":
        left_hand_array = Array('d', 75, lock = True)          # [input]
        right_hand_array = Array('d', 75, lock = True)         # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, retargeting_method=args.retargeting_method)
    else:
        pass
    
    if args.record:
        recorder = EpisodeWriter(task_dir = args.task_dir, frequency = args.frequency, rerun_log = True)
        recording = False
        logger.info(f"Episode recorder initialized with task_dir={args.task_dir}") 
    try:
        user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        if user_input.lower() == 'r':
            if not args.no_gradual_speed:
                arm_ctrl.speed_gradual_max()
                logger.info("Gradual speed increase enabled")
                
            # Enable head tracking for active camera if available
            if camera_controller and camera_controller.connected:
                try:
                    logger.info("Enabling head tracking for active camera...")
                    if camera_controller.enable_head_tracking(tv_wrapper):
                        logger.info("Head tracking enabled successfully")
                    else:
                        logger.warning("Failed to enable head tracking")
                except Exception as e:
                    logger.error(f"Error enabling head tracking: {e}")
            
            running = True
            frame_counter = 0
            
            logger.info("Starting main control loop")
            while running:
                start_time = time.time()
                
                # Get pose data
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
                frame_counter += 1
                
                # Log pose data if enabled
                if pose_logger:
                    pose_logger.log_pose(head_rmat, left_wrist, right_wrist, left_hand, right_hand)
                
                # Active camera control is now handled by the threaded controller
                # Just get current servo states for recording if needed
                camera_servo_states = None
                if camera_controller and camera_controller.connected and camera_controller.head_tracking_enabled:
                    try:
                        camera_servo_states = camera_controller.get_servo_states()
                    except Exception as e:
                        logger.warning(f"Error reading active camera servo states: {e}")

                # send hand skeleton data to hand_ctrl.control_process
                if args.hand:
                    left_hand_array[:] = left_hand.flatten()
                    right_hand_array[:] = right_hand.flatten()

                # get current state data.
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                # print(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

                # Log periodic IK timing information
                if frame_counter % 300 == 0:  # Every ~10 seconds at 30fps
                    logger.info(f"IK solve time: {(time_ik_end - time_ik_start)*1000:.2f}ms")

                tv_resized_image = cv2.resize(vr_img_array, (vr_img_shape[1] // 2, vr_img_shape[0] // 2))
                
                # Add recording status overlay
                if args.record:
                    status_text = ""
                    help_text = ""
                    if recording:
                        status_text = "RECORDING"
                        help_text = "Controls: [r]=abort | [q]=save optimal | [w]=save suboptimal | [e]=save recovery"
                    else:
                        status_text = "READY TO RECORD"
                        help_text = "Controls: [s]=start recording | [q]=quit (no recording) | [ESC]=quit"
                    
                    # Add status text overlay
                    cv2.putText(tv_resized_image, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (0, 255, 0), 2)
                    
                    # Add help text overlay
                    cv2.putText(tv_resized_image, help_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                
                # Handle key presses for recording control
                if key == 27 and not args.record:
                    running = False
                    logger.info("User requested exit with 'ESC' key")
                elif args.record:
                    if key == ord('s') and not recording:
                        # Start recording
                        if recorder.create_episode():
                            recording = True
                            logger.info("Started recording episode")
                        else:
                            logger.warning("Failed to create recording episode")
                    elif key == ord('r') and recording:
                        # Abort recording
                        recorder.abort_episode()
                        recording = False
                        logger.info("Aborted recording episode")
                    elif recording and key in [ord('q'), ord('w'), ord('e')]:
                        # Save with quality labels
                        quality_map = {ord('q'): 'optimal', ord('w'): 'suboptimal', ord('e'): 'recovery'}
                        quality = quality_map[key]
                        recorder.save_episode(quality=quality)
                        recording = False
                        logger.info(f"Saved recording episode with quality: {quality}")

                # record data
                if args.record:
                    # dex hand or gripper
                    if args.hand == "dex3":
                        with dual_hand_data_lock:
                            if args.force:
                                # [q0...q6, dq0...dq6, tau0...tau6, p0...p11] (33 each)
                                left_hand_state = dual_hand_state_array[0:7]
                                left_hand_vel = dual_hand_state_array[7:14]
                                left_hand_torque = dual_hand_state_array[14:21]
                                left_hand_pressures = dual_hand_state_array[21:33]
                                right_hand_state = dual_hand_state_array[33:40]
                                right_hand_vel = dual_hand_state_array[40:47]
                                right_hand_torque = dual_hand_state_array[47:54]
                                right_hand_pressures = dual_hand_state_array[54:66]
                            else:
                                # [q0...q6, p0...p11] (19 each)
                                left_hand_state = dual_hand_state_array[0:7]
                                left_hand_vel = []  # Not available without --force
                                left_hand_torque = []  # Not available without --force
                                left_hand_pressures = dual_hand_state_array[7:19]
                                right_hand_state = dual_hand_state_array[19:26]
                                right_hand_vel = []  # Not available without --force
                                right_hand_torque = []  # Not available without --force
                                right_hand_pressures = dual_hand_state_array[26:38]
                            left_hand_action = dual_hand_action_array[:7]
                            right_hand_action = dual_hand_action_array[-7:]
                    elif args.hand == "gripper":
                        with dual_gripper_data_lock:
                            left_hand_state = [dual_gripper_state_array[1]]
                            right_hand_state = [dual_gripper_state_array[0]]
                            left_hand_action = [dual_gripper_action_array[1]]
                            right_hand_action = [dual_gripper_action_array[0]]
                            # No pressure sensors for gripper
                            left_hand_pressures = []
                            right_hand_pressures = []
                            # Add velocities and torques for gripper
                            left_hand_vel = []
                            right_hand_vel = []
                            left_hand_torque = []
                            right_hand_torque = []
                    elif args.hand == "inspire1":
                        with dual_hand_data_lock:
                            left_hand_state = dual_hand_state_array[:6]
                            right_hand_state = dual_hand_state_array[-6:]
                            left_hand_action = dual_hand_action_array[:6]
                            right_hand_action = dual_hand_action_array[-6:]
                            # No pressure sensors for inspire hand
                            left_hand_pressures = []
                            right_hand_pressures = []
                            # Add velocities and torques for inspire hand
                            left_hand_vel = []
                            right_hand_vel = []
                            left_hand_torque = []
                            right_hand_torque = []
                    else:
                        print("No dexterous hand set.")
                        pass
                    # head image
                    current_head_cam_image = head_cam_img_array.copy()
                    # active camera image (separate from recording)
                    if args.active_camera and active_cam_img_array is not None:
                        current_active_cam_image = active_cam_img_array.copy()
                    # wrist image
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()
                    # arm state and action
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]
                    # Get velocities and torques for arms
                    left_arm_vel  = current_lr_arm_dq[:7] if (args.force and len(current_lr_arm_dq) >= 14) else []
                    right_arm_vel = current_lr_arm_dq[-7:] if (args.force and len(current_lr_arm_dq) >= 14) else []
                    # Try to get torques if available
                    try:
                        current_lr_arm_torque = arm_ctrl.get_current_dual_arm_torque()
                        left_arm_torque = current_lr_arm_torque[:7] if current_lr_arm_torque is not None else []
                        right_arm_torque = current_lr_arm_torque[-7:] if current_lr_arm_torque is not None else []
                    except Exception:
                        left_arm_torque = []
                        right_arm_torque = []
                    
                    # Active camera servo data
                    if camera_servo_states:
                        camera_current_pitch = camera_servo_states['current_pitch']
                        camera_current_yaw = camera_servo_states['current_yaw']
                        camera_target_pitch = camera_servo_states['target_pitch']
                        camera_target_yaw = camera_servo_states['target_yaw']
                    else:
                        # No camera data available
                        camera_current_pitch = 0.0
                        camera_current_yaw = 0.0
                        camera_target_pitch = 0.0
                        camera_target_yaw = 0.0
                    
                    # Note: Hand torque and velocity values are already collected above in the hand-specific sections

                    if recording:
                        colors = {}
                        depths = {}
                        
                        if args.active_camera:
                            # Active camera mode: split the downscaled active camera image (480x1280) into left/right (480x640 each)

                            colors[f"color_{4}"] = current_active_cam_image[:, :640]   # color_4.jpg - left active camera (480x640)
                            colors[f"color_{5}"] = current_active_cam_image[:, 640:]   # color_5.jpg - right active camera (480x640)
                            
                            # Head camera from recording stream (same as active camera for dataset compatibility)
                            if BINOCULAR:
                                colors[f"color_{0}"] = current_head_cam_image[:, :head_cam_img_shape[1]//2]  # Left eye head camera
                                colors[f"color_{1}"] = current_head_cam_image[:, head_cam_img_shape[1]//2:]  # Right eye from head camera
                            else:
                                colors[f"color_{0}"] = current_head_cam_image
                            
                            # Wrist cameras
                            if WRIST:
                                colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]  # Left wrist
                                colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]  # Right wrist
                        else:
                            # Head camera mode: use traditional layout
                            if BINOCULAR:
                                colors[f"color_{0}"] = current_head_cam_image[:, :head_cam_img_shape[1]//2]
                                colors[f"color_{1}"] = current_head_cam_image[:, head_cam_img_shape[1]//2:]
                                if WRIST:
                                    colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                    colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                            else:
                                colors[f"color_{0}"] = current_head_cam_image
                                if WRIST:
                                    colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                    colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        states = {
                            "left_arm": {                                                                    
                                "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                                "qvel":   left_arm_vel if isinstance(left_arm_vel, list) else left_arm_vel.tolist(),
                                "torque": left_arm_torque if isinstance(left_arm_torque, list) else left_arm_torque.tolist(),                        
                            }, 
                            "right_arm": {                                                                    
                                "qpos":   right_arm_state.tolist(),       
                                "qvel":   right_arm_vel if isinstance(right_arm_vel, list) else right_arm_vel.tolist(),
                                "torque": right_arm_torque if isinstance(right_arm_torque, list) else right_arm_torque.tolist(),                         
                            },                        
                            "left_hand": {                                                                    
                                "qpos":   left_hand_state,           
                                "qvel":   left_hand_vel if isinstance(left_hand_vel, list) else left_hand_vel,                          
                                "torque": left_hand_torque if isinstance(left_hand_torque, list) else left_hand_torque,                          
                                "pressures": left_hand_pressures,
                            }, 
                            "right_hand": {                                                                    
                                "qpos":   right_hand_state,       
                                "qvel":   right_hand_vel if isinstance(right_hand_vel, list) else right_hand_vel,                          
                                "torque": right_hand_torque if isinstance(right_hand_torque, list) else right_hand_torque, 
                                "pressures": right_hand_pressures,
                            }, 
                            "body": None, # TODO Hier könnte man um den Körper Erweitern
                        }
                        
                        # Add camera servo states if active camera is enabled
                        if args.active_camera and camera_servo_states:
                            states["camera"] = {
                                "qpos": [camera_current_pitch, camera_current_yaw],  # Current positions in radians
                                "qvel": [],  # Velocity not available
                                "torque": []  # Torque not available
                            }
                        else:
                            # No camera data available
                            states["camera"] = {
                                "qpos": [],
                                "qvel": [],
                                "torque": []
                            }
                        
                        actions = { # Todo torque und pressure auch als action? Laut Paper bi-ACT schon.
                            "left_arm": {                                   
                                "qpos":   left_arm_action.tolist(),       
                                "qvel":   [],       
                                "torque": [],      
                            }, 
                            "right_arm": {                                   
                                "qpos":   right_arm_action.tolist(),       
                                "qvel":   [],       
                                "torque": [],       
                            },                         
                            "left_hand": {                                   
                                "qpos":   left_hand_action,       
                                "qvel":   [],       
                                "torque": [],       
                            }, 
                            "right_hand": {                                   
                                "qpos":   right_hand_action,       
                                "qvel":   [],       
                                "torque": [], 
                            }, 
                            "body": None, 
                        }
                        
                        # Add camera servo actions if active camera is enabled
                        if args.active_camera and camera_servo_states:
                            actions["camera"] = {
                                "qpos": [camera_target_pitch, camera_target_yaw],  # Target positions in radians
                                "qvel": [],
                                "torque": []
                            }
                        else:
                            # No camera data available
                            actions["camera"] = {
                                "qpos": [],
                                "qvel": [],
                                "torque": []
                            }
                        
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)

                # Only log performance details occasionally to avoid flooding the log
                if frame_counter % 300 == 0:  # Every ~10 seconds at 30fps
                    actual_frequency = 1.0 / (time_elapsed + sleep_time) if (time_elapsed + sleep_time) > 0 else args.frequency
                    logger.info(f"Performance: frame_time={time_elapsed*1000:.1f}ms, sleep={sleep_time*1000:.1f}ms, actual_freq={actual_frequency:.1f}Hz")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    finally:
        # Clean up
        if pose_logger:
            pose_logger.stop()
            logger.info("Stopped pose logger and saved data")
            
        # Cleanup camera controller
        if camera_controller:
            camera_controller.disconnect()
            logger.info("Active camera controller disconnected")
            
        arm_ctrl.ctrl_dual_arm_go_home()
        logger.info("Arms returned to home position")
        
        vr_img_shm.unlink()
        vr_img_shm.close()
        head_cam_img_shm.unlink()
        head_cam_img_shm.close()
        if args.active_camera:
            active_cam_img_shm.unlink()
            active_cam_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()
        if args.record:
            recorder.close()
        logger.info("Resources cleaned up, exiting program")
        exit(0)