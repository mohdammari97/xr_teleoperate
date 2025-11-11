import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Value, Array, Lock
import threading
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller,Dex3_1_Controller_console
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.robot_control.robot_hand_brainco import Brainco_Controller
from teleop.robot_control.active_head_cam import ActiveCameraController
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
#from teleop.utils.ipc import IPC_Server
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int,publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
start_signal = False
running = True
should_toggle_recording = False
is_recording = False
def on_press(key):
    global running, start_signal, should_toggle_recording
    if key == 'r':
        start_signal = True
        logger_mp.info("Program start signal received.")
    elif key == 'q' and start_signal == True:
        stop_listening()
        running = False
    elif key == 's' and start_signal == True:
        should_toggle_recording = True
    else:
        logger_mp.info(f"{key} was pressed, but no action is defined for this key.")
listen_keyboard_thread = threading.Thread(target=listen_keyboard, kwargs={"on_press": on_press, "until": None, "sequential": False,}, daemon=True)
listen_keyboard_thread.start()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--frequency', type = float, default = 20.0, help = 'save data\'s frequency')

    # basic control parameters
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    # mode flags
    
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task name for recording')
    parser.add_argument('--task-goal', type = str, default = 'e.g. pick the red cube on the table.', help = 'task goal for recording')
    # Active Camera options
    parser.add_argument('--use-active-cam', action='store_true', default=False, help='Enable active camera head tracking')
    parser.add_argument('--camera-port', type=str, default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3R4A5A-if00-port0", 
                       help='Serial port for the active camera servo controller')
    parser.add_argument('--camera-safe-mode', action='store_true', default=False, help='Enable safe mode with limited camera movement')
    parser.add_argument('--camera-max-movement', type=float, default=60.0, help='Maximum camera movement in degrees from start position')

    # Speed Limit
    parser.add_argument('--arm-speed', type=float, default=10.0, 
                      help='Set the arm velocity limit (default is controller-specific)')
    parser.add_argument('--no-gradual-speed', action='store_true',
                      help='Disable gradual speed increase')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    if args.sim:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [720, 1280],  # Head camera resolution
            'head_camera_id_numbers': [6],
            #'wrist_camera_type': 'opencv',
            #'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
            #'wrist_camera_id_numbers': [2, 4],
        }
    else:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 1280],  # Head camera resolution
            'head_camera_id_numbers': [6],
            'wrist_camera_type': 'opencv',
            'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
            'wrist_camera_id_numbers': [2, 4],
        }
        # Add active camera config if enabled
        if args.use_active_cam:
            img_config.update({
                'active_camera_type': 'opencv',
                'active_camera_image_shape': [720, 2560],  # Resolution of active cam
                'active_camera_id_numbers': [12],
            })


    # Configure VR display resolutions based on active camera selection
    if args.use_active_cam:
        # Use active camera for VR headset with full resolution
        active_cam_img_shape = (img_config['active_camera_image_shape'][0], img_config['active_camera_image_shape'][1], 3)  # 720x2560 for VR
        logger_mp.info("Using active camera for VR headset with full resolution (720x2560)")
    else:
        # Use head camera
        active_cam_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)  # Standard head camera resolution for VR
        logger_mp.info("Using head camera for VR headset")

    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    # Add shared memory for active camera recording if enabled
    if args.use_active_cam:
        active_cam_img_shape = (720, 2560, 3)  # Recording resolution for active cam
        # Set up shared memory for VR display (full resolution or cropped)
        if BINOCULAR and not (active_cam_img_shape[1] / active_cam_img_shape[0] > ASPECT_RATIO_THRESHOLD):
            active_cam_img_shape = (active_cam_img_shape[0], active_cam_img_shape[1] * 2, 3)
        active_cam_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(active_cam_img_shape) * np.uint8().itemsize)
        active_cam_img_array = np.ndarray(active_cam_img_shape, dtype = np.uint8, buffer = active_cam_img_shm.buf)

    if WRIST and args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name, server_address="127.0.0.1")
    elif WRIST and not args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        if args.use_active_cam:
            img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                     wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name,
                                     active_cam_img_shape = active_cam_img_shape, active_cam_img_shm_name = active_cam_img_shm.name,
                                     use_active_camera = True)
        else:
            img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                     wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        if args.use_active_cam:
            img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name,
                                     active_cam_img_shape = active_cam_img_shape, active_cam_img_shm_name = active_cam_img_shm.name,
                                     use_active_camera = True)
        else:
            img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=active_cam_img_shape, img_shm_name=active_cam_img_shm.name, 
                                 return_state_data=True, return_hand_rot_data = False)

    # arm
    if args.arm == "G1_29":
        arm_ik = G1_29_ArmIK()
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        if args.arm_speed is not None:
            arm_ctrl.arm_velocity_limit = args.arm_speed
            logger_mp.info(f"Setting custom arm velocity limit: {args.arm_speed}")
    elif args.arm == 'G1_23':
        arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        arm_ik = G1_23_ArmIK()
    elif args.arm == "H1_2":
        arm_ctrl = H1_2_ArmController(simulation_mode=args.sim)
        arm_ik = H1_2_ArmIK()
    elif args.arm == "H1":
        arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        arm_ik = H1_ArmIK()
    # active camera
    camera_controller = None
    if args.use_active_cam:
        try:
            logger_mp.info("Initializing active camera controller...")
            camera_controller = ActiveCameraController(
                port=args.camera_port,
                safe_mode=args.camera_safe_mode,
                max_movement_deg=args.camera_max_movement,
                logger=logger_mp
            )
            
            # Connect to servos
            logger_mp.info("Connecting to camera servos...")
            if not camera_controller.connect():
                logger_mp.error("Failed to connect to servos")
                camera_controller = None
        except Exception as e:
            logger_mp.error(f"Failed to initialize active camera: {e}")
            camera_controller = None

    # end-effector
    if args.ee == "dex3" and args.xr_mode == "hand":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    elif args.ee == "dex3" and args.xr_mode == "controller":
        left_hand_value_in = Value('d', 0.0, lock=True)      # [input]
        right_hand_value_in = Value('d', 0.0, lock=True)     # [input]
        left_aButton_in  = Value('b', False, lock=True)         
        left_bButton_in = Value('b', False, lock=True)
        right_aButton_in = Value('b', False, lock=True)
        right_bButton_in = Value('b', False, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller_console(left_hand_value_in, right_hand_value_in, left_aButton_in,left_bButton_in,right_aButton_in,right_bButton_in,
                                                  dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
    elif args.ee == "inspire1":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        hand_ctrl = Inspire_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    elif args.ee == "brainco":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    else:
        pass

    # simulation mode
    if args.sim:
        reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        reset_pose_publisher.Init()
        from teleop.utils.sim_state_topic import start_sim_state_subscribe
        sim_state_subscriber = start_sim_state_subscribe()

    # controller + motion mode
    if args.xr_mode == "controller" and args.motion:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        sport_client = LocoClient()
        sport_client.SetTimeout(0.0001)
        sport_client.Init()
    
    # record + headless mode
    if args.record and args.headless:
        recorder = EpisodeWriter(task_dir = args.task_dir + args.task_name, task_goal = args.task_goal, frequency = args.frequency, rerun_log = False)
    elif args.record and not args.headless:
        recorder = EpisodeWriter(task_dir = args.task_dir + args.task_name, task_goal = args.task_goal, frequency = args.frequency, rerun_log = True)
    episode_start_position = None  # Store position at episode start
    try:
        logger_mp.info("Please enter the start signal (enter 'r' to start the subsequent program)")
        while not start_signal:
            time.sleep(0.01)
        
        if not args.no_gradual_speed:
            arm_ctrl.speed_gradual_max()
            logger_mp.info("Gradual speed increase enabled")
        else:
            logger_mp.info("Gradual speed increase disabled")
        
        # Enable head tracking if active camera is available
        if camera_controller:
            logger_mp.info("Enabling head tracking...")
            if not camera_controller.enable_head_tracking(tv_wrapper):
                logger_mp.error("Failed to enable head tracking")
                camera_controller = None
            else:
                logger_mp.info("Head tracking enabled! Camera will follow head movements automatically.")
        
        while running:
            start_time = time.time()

            if not args.headless:
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_listening()
                    running = False
                    if args.sim:
                        publish_reset_category(2, reset_pose_publisher)
                elif key == ord('s'):
                    should_toggle_recording = True
                elif key == ord('a'):
                    if args.sim:
                        publish_reset_category(2, reset_pose_publisher)

            if args.record and should_toggle_recording:
                should_toggle_recording = False
                if not is_recording:
                    if recorder.create_episode():
                        is_recording = True
                        # Reset position reference at episode start
                        episode_start_position = arm_ctrl.get_current_robot_position().copy()
                        logger_mp.info(f"Episode started - position offset set to: {episode_start_position}")
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    is_recording = False
                    recorder.save_episode()
                    logger_mp.info("Episode saved. Robot position will be reset when next episode starts.")
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
            # get input data
            tele_data = tv_wrapper.get_motion_state_data()
            if (args.ee == "dex3" or args.ee == "inspire1" or args.ee == "brainco") and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()

            elif args.ee == "dex3" and args.xr_mode == "controller":
                with left_hand_value_in.get_lock():
                    left_hand_value_in.value = tele_data.left_trigger_value
                with right_hand_value_in.get_lock():
                    right_hand_value_in.value = tele_data.right_trigger_value
                with left_aButton_in.get_lock():
                    left_aButton_in.value = tele_data.tele_state.left_aButton  # True / False
                with left_bButton_in.get_lock():
                    left_bButton_in.value = tele_data.tele_state.left_bButton
                with right_aButton_in.get_lock():
                    right_aButton_in.value = tele_data.tele_state.right_aButton
                with right_bButton_in.get_lock():
                    right_bButton_in.value = tele_data.tele_state.right_bButton
                    
            elif args.ee == "dex1" and args.xr_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_trigger_value
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_trigger_value
            elif args.ee == "dex1" and args.xr_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_pinch_value
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_pinch_value
            else:
                pass        
            
            # high level control
            if args.xr_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.tele_state.right_aButton and tele_data.tele_state.right_bButton:
                    stop_listening()
                    running = False
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.tele_state.left_thumbstick_state and tele_data.tele_state.right_thumbstick_state:
                    sport_client.Damp()
                # control, limit velocity to within 0.3
                sport_client.Move(-tele_data.tele_state.left_thumbstick_value[1]  * 0.6,
                                  -tele_data.tele_state.left_thumbstick_value[0]  * 0.6,
                                  -tele_data.tele_state.right_thumbstick_value[0] * 0.6)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # get current robot velocity data from odometry subscriber
            robot_vel = arm_ctrl.get_current_robot_velocity()
            # get current robot position data from odometry subscriber
            raw_robot_pos = arm_ctrl.get_current_robot_position()

            controller_combination = [
                tele_data.left_trigger_value,
                tele_data.right_trigger_value,
                tele_data.tele_state.left_aButton,
                tele_data.tele_state.left_bButton,
                tele_data.tele_state.right_aButton,
                tele_data.tele_state.right_bButton,
                tele_data.tele_state.left_thumbstick_value,
                tele_data.tele_state.right_thumbstick_value,
                tele_data.tele_state.left_squeeze_ctrl_value,
                tele_data.tele_state.right_squeeze_ctrl_value,
            ]

            if episode_start_position is not None:
                robot_pos = [
                    raw_robot_pos[0] - episode_start_position[0],
                    raw_robot_pos[1] - episode_start_position[1],
                    raw_robot_pos[2] - episode_start_position[2]
                ]
            else:
                robot_pos = raw_robot_pos
            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_arm_pose, tele_data.right_arm_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            #robot_vel_action = arm_ctrl.get_velocity_commands() #unitree controller
            robot_vel_action = [-tele_data.tele_state.left_thumbstick_value[1]  * 0.6, -tele_data.tele_state.left_thumbstick_value[0]  * 0.6, -tele_data.tele_state.right_thumbstick_value[0]  * 0.6] #metaquest controller
            #print(robot_vel_action)
            camera_servo_states = None
            if camera_controller and camera_controller.connected and camera_controller.head_tracking_enabled:
                try:
                    camera_servo_states = camera_controller.get_servo_states()
                except Exception as e:
                    logger_mp.warning(f"Error reading active camera servo states: {e}")

            
            # record data
            if args.record:
                # dex hand or gripper
                if args.ee == "dex3" and args.xr_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []

                elif args.ee == "dex3" and args.xr_mode == "controller":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []

                elif args.ee == "dex1" and args.xr_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.xr_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.tele_state.left_thumbstick_value[1]  * 0.3,
                                               -tele_data.tele_state.left_thumbstick_value[0]  * 0.3,
                                               -tele_data.tele_state.right_thumbstick_value[0] * 0.3]
                elif (args.ee == "inspire1" or args.ee == "brainco") and args.xr_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []
                # head image
                current_tv_image = tv_img_array.copy()
                
                # active camera image (for recording)
                if args.use_active_cam:
                    current_active_cam_image = active_cam_img_array.copy()
                # wrist image
                if WRIST:
                    current_wrist_image = wrist_img_array.copy()
                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                
  
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
                if is_recording:
                    colors = {}
                    depths = {}
                    
                    
                    if args.use_active_cam:
                        # Save ONLY active camera + wrist cameras
                        # Split active cam (stereo side-by-side) into left/right
                        half_w = current_active_cam_image.shape[1] // 2
                        left_active  = current_active_cam_image[:, :half_w]
                        right_active = current_active_cam_image[:, half_w:]

                        # Force old resolution (480x640 per half) regardless of SHM source size
                        left_active  = cv2.resize(left_active,  (640, 480))
                        right_active = cv2.resize(right_active, (640, 480))

                        # Save active cams as color_0/1 (you asked to renumber to 0/1)
                        colors["color_0"] = left_active
                        colors["color_1"] = right_active
                        colors["color_4"] = current_tv_image[:, :tv_img_shape[1]//2]  # head camera left
                        colors["color_5"] = current_tv_image[:, tv_img_shape[1]//2:]

                        # Wrist cameras (if present), split left/right
                        if WRIST:
                            half_w_wrist = wrist_img_shape[1] // 2
                            colors["color_2"] = current_wrist_image[:, :half_w_wrist]   # wrist left
                            colors["color_3"] = current_wrist_image[:, half_w_wrist:]   # wrist right

                        # IMPORTANT: do NOT add color_0/color_1 (head/TV) here
                    else:
                        # No active camera -> keep head/TV (for completeness)
                        if BINOCULAR:
                            colors["color_0"] = current_tv_image[:, :tv_img_shape[1]//2]
                            colors["color_1"] = current_tv_image[:, tv_img_shape[1]//2:]
                            if WRIST:
                                half_w_wrist = wrist_img_shape[1] // 2
                                colors["color_2"] = current_wrist_image[:, :half_w_wrist]
                                colors["color_3"] = current_wrist_image[:, half_w_wrist:]
                        else:
                            colors["color_0"] = current_tv_image
                            if WRIST:
                                half_w_wrist = wrist_img_shape[1] // 2
                                colors["color_1"] = current_wrist_image[:, :half_w_wrist]
                                colors["color_2"] = current_wrist_image[:, half_w_wrist:]
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body_vel": {
                            "qpos": [],
                            "qvel": robot_vel if isinstance(robot_vel, list) else robot_vel.tolist(),
                            "torque": [],
                        }, 
                        "odometry": {
                            "qpos": robot_pos if isinstance(robot_pos, list) else robot_pos.tolist(), 
                            "qvel": [],
                            "torque": [],
                        },
                        "controller": {
                            "qpos": [], #controller_combination if isinstance(controller_combination, list) else controller_combination.tolist(), 
                            "qvel": [],
                            "torque": [],
                        },

                        
                    }
                    if args.use_active_cam and camera_servo_states:
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
                    actions = {
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
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body_vel": {
                            "qpos": [], 
                            "qvel": robot_vel_action if isinstance(robot_vel_action, list) else robot_vel_action.tolist(),
                            "torque": [],
                        },
                        "odometry": {
                            "qpos": [], #keep empty for now
                            "qvel": [],
                            "torque": [],
                        },
                        
                    }
                    if args.use_active_cam and camera_servo_states:
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
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        if args.sim:
            sim_state_subscriber.stop_subscribe()
        if camera_controller:
            camera_controller.disconnect()
            logger_mp.info("Active camera controller disconnected")
        tv_img_shm.close()
        tv_img_shm.unlink()
        if args.use_active_cam:
            active_cam_img_shm.close()
            active_cam_img_shm.unlink()
        if WRIST:
            wrist_img_shm.close()
            wrist_img_shm.unlink()
        if args.record:
            recorder.close()
        listen_keyboard_thread.join()
        logger_mp.info("Finally, exiting program...")
        exit(0)
