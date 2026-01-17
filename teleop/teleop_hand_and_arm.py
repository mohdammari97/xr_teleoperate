import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Value, Array, Lock
import threading
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)
from scipy.spatial.transform import Rotation as R


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
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.robot_control.robot_hand_brainco import Brainco_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int,publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion  
STOP           = False  # Enable to begin system exit procedure
RECORD_TOGGLE  = False  # [Ready] ⇄ [Recording] ⟶ [AutoSave] ⟶ [Ready]         (⇄ manual) (⟶ auto)
RECORD_RUNNING = False  # True if [Recording]
RECORD_READY   = True   # True if [Ready], False if [Recording] / [AutoSave]
# task info
TASK_NAME = None
TASK_DESC = None
ITEM_ID = None
# camera head offset (for active camera control)
prev_dx, prev_dy = 0.0, 0.0

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def on_info(info):
    """Only handle CMD_TOGGLE_RECORD's task info"""
    global TASK_NAME, TASK_DESC, ITEM_ID
    TASK_NAME   = info.get("task_name")
    TASK_DESC   = info.get("task_desc")
    ITEM_ID     = info.get("item_id")
    logger_mp.debug(f"[on_info] Updated globals: {TASK_NAME}, {TASK_DESC}, {ITEM_ID}")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, RECORD_READY
    return {
        "START": START,
        "STOP": STOP,
        "RECORD_RUNNING": RECORD_RUNNING,
        "RECORD_READY": RECORD_READY,
    }


def quat_to_offset(quat, max_angle=(30, 20)):
    """
    Convert head quaternion to screen offset (-1~1 for dx, dy)
    dx: left-right (yaw)
    dy: up-down (pitch)
    """
    if quat is None or len(quat) != 4:
        logger_mp.warning(f"Invalid quat: {quat}, using default [0,0,0,1]")
        quat = [0, 0, 0, 1]

    r = R.from_quat(quat)

    # Important: VR head often uses yaw-pitch-roll = (Y, X, Z) order
    yaw, pitch, roll = r.as_euler('yxz', degrees=True)

    # Mapping:
    # yaw → left/right
    # pitch → up/down
    dx = np.clip(-yaw / max_angle[0], -1, 1)     # 左右转头
    dy = np.clip(-pitch / max_angle[1], -1, 1)  # 点头，上下
    
    return dx, dy




def update_move_image_from_tv(tv_img_array, move_image_img_array, dx, dy):
    """
    Update move_image_img_array based on head offsets (dx, dy)
    Works for binocular images:
        - tv_img_array: (H, 2W, 3)
        - move_image_img_array: (h, 2w, 3)
    """
    H, W2, _ = tv_img_array.shape
    h, w2, _ = move_image_img_array.shape
    W = W2 // 2        # each eye width
    w = w2 // 2        # each eye width

    # Split tv image into left-eye and right-eye images
    tv_left  = tv_img_array[:, :W, :]
    tv_right = tv_img_array[:, W:, :]

    # Output placeholders
    out_left  = np.zeros((h, w, 3), dtype=np.uint8)
    out_right = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert dx dy to pixel offsets (use same mapping for both eyes)
    x0 = int((W - w) * (dx + 1) / 2)
    y0 = int((H - h) * (dy + 1) / 2)

    x0 = np.clip(x0, 0, W - w)
    y0 = np.clip(y0, 0, H - h)

    # Crop left eye
    sub_left = tv_left[y0:y0+h, x0:x0+w, :]
    # Crop right eye
    sub_right = tv_right[y0:y0+h, x0:x0+w, :]

    np.copyto(out_left, sub_left)
    np.copyto(out_right, sub_right)

    # Stitch back into the output image
    move_image_img_array[:, :w, :] = out_left
    move_image_img_array[:, w:, :] = out_right# Copy to output
    


def teledata_to_list(tele_data):
    # 6 parameters  
    result = []
    # result.extend(tele_data.head_pose.flatten().tolist())

    for val in [
        # tele_data.left_pinch_value,
        # tele_data.right_pinch_value,
        tele_data.left_trigger_value,
        tele_data.right_trigger_value
    ]:
        result.append(0.0 if val is None else val)


    s = tele_data.tele_state


    # bools = [
    #     s.left_pinch_state, s.left_squeeze_state,
    #     s.right_pinch_state, s.right_squeeze_state,
    #     s.left_trigger_state, s.left_squeeze_ctrl_state,
    #     s.left_thumbstick_state, s.left_aButton, s.left_bButton,
    #     s.right_trigger_state, s.right_squeeze_ctrl_state,
    #     s.right_thumbstick_state, s.right_aButton, s.right_bButton,
    # ]
    bools = [
        s.left_aButton, s.left_bButton,
        s.right_aButton, s.right_bButton,
    ]
    result.extend([float(b) for b in bools])

    # result.extend(s.left_thumbstick_value.tolist())
    # result.extend(s.right_thumbstick_value.tolist())

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'save data\'s frequency')

    # basic control parameters
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording')
    parser.add_argument('--task-dir', type = str, default = '/media/mohdammari97/PortableSSD/standing_brown_bag/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'standing_brown_bag', help = 'task name for recording')
    parser.add_argument('--task-desc', type = str, default = 'Pick up one brown bag and put it on the metal tray.', help = 'task goal for recording')

    parser.add_argument('--move-image', action='store_true', default=False, help='Enable active image head tracking')
    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    try:
        # ipc communication. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press, on_info=on_info, get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, kwargs={"on_press": on_press, "until": None, "sequential": False,}, daemon=True)
            listen_keyboard_thread.start()

        # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
        if args.sim:
            img_config = {
                'fps': 30,
                'head_camera_type': 'opencv',
                'head_camera_image_shape': [480, 640],  # Head camera resolution
                'head_camera_id_numbers': [0],
                'wrist_camera_type': 'opencv',
                'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
                'wrist_camera_id_numbers': [2, 4],
            }
        else:
            img_config = {
                'fps': 30,
                'head_camera_type': 'opencv',
                'head_camera_image_shape': [480, 1280],  # Head camera resolution
                'head_camera_id_numbers': [0],
                'wrist_camera_type': 'opencv',
                'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
                'wrist_camera_id_numbers': [2, 4],
            }
        if args.move_image:
            img_config.update({
                'move_image_type': 'opencv',
                'move_image_shape': [240, 640],  # Resolution of active cam
                'move_image_id_numbers': [5,6],
            })

        if args.move_image:
            # Use active camera for VR headset with full resolution
            move_image_img_shape = (img_config['move_image_shape'][0], img_config['move_image_shape'][1], 3)  # 720x2560 for VR
            logger_mp.info("Using move_image for VR headset with full resolution [240, 640]")
        else:
            # Use head camera
            move_image_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)  # Standard head camera resolution for VR
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
        
        if args.move_image:
            # if BINOCULAR and not (move_image_img_shape[1] / move_image_img_shape[0] > ASPECT_RATIO_THRESHOLD):
            #     move_image_img_shape = (move_image_img_shape[0], move_image_img_shape[1] * 2, 3)
            move_image_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(move_image_img_shape) * np.uint8().itemsize)
            move_image_img_array = np.ndarray(move_image_img_shape, dtype = np.uint8, buffer = move_image_img_shm.buf)

        # 1. 创建共享内存
        tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
        tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)
        #如果有手腕相机（WRIST），也会同样创建共享内存：
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
            img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                    wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
        else:
            img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)
        #图像由 ImageClient 线程接收：
        image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
        image_receive_thread.daemon = True
        image_receive_thread.start()

        # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        # TeleVuerWrapper 是 XR 显示端接口，它通过 共享内存名字 tv_img_shm.name 来访问图像。
        tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=move_image_img_shape, img_shm_name=move_image_img_shm.name, 
                                    return_state_data=True, return_hand_rot_data = False)
        
        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

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
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20) # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
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
            recorder = EpisodeWriter(task_dir = args.task_dir + args.task_name, task_goal = args.task_desc, frequency = args.frequency, rerun_log = False)
        elif args.record and not args.headless:
            recorder = EpisodeWriter(task_dir = args.task_dir + args.task_name, task_goal = args.task_desc, frequency = args.frequency, rerun_log = True)
        episode_start_position = None  

        logger_mp.info("Please enter the start signal (enter 'r' to start the subsequent program)")
        while not START and not STOP:
            time.sleep(0.01)
        logger_mp.info("start program.")
        arm_ctrl.speed_gradual_max()
        while not STOP:
            start_time = time.time()

            if not args.headless:
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)
                # opencv GUI communication
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    START = False
                    STOP = True
                    if args.sim:
                        publish_reset_category(2, reset_pose_publisher)
                elif key == ord('s'):
                    RECORD_TOGGLE = True
                elif key == ord('a'):
                    if args.sim:
                        publish_reset_category(2, reset_pose_publisher)

            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                        # Reset position reference at episode start
                        episode_start_position = arm_ctrl.get_current_robot_position().copy()
                        logger_mp.info(f"Episode started - position offset set to: {episode_start_position}")
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
            # get input data
            tele_data = tv_wrapper.get_motion_state_data()
            quat = tv_wrapper.get_head_orientation()
            dx, dy=quat_to_offset(quat)

            # camera offset action (delta from previous step)
            delta_dx = dx - prev_dx
            delta_dy = dy - prev_dy
            prev_dx, prev_dy = dx, dy

            update_move_image_from_tv(tv_img_array, move_image_img_array, dx, dy)
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
                # if tele_data.tele_state.left_thumbstick_state and tele_data.tele_state.right_thumbstick_state:
                #     sport_client.Damp()
                # control, limit velocity to within 0.3
                sport_client.Move(-tele_data.tele_state.left_thumbstick_value[1]  * 0.5,
                                  -tele_data.tele_state.left_thumbstick_value[0]  * 0.5,
                                  -tele_data.tele_state.right_thumbstick_value[0] * 0.5)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # get current robot velocity data from odometry subscriber
            robot_vel = arm_ctrl.get_current_robot_velocity()
            # get current robot position data from odometry subscriber
            raw_robot_pos = arm_ctrl.get_current_robot_position()

            controller_combination = teledata_to_list(tele_data)

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
            

            # record data
            if args.record:
                RECORD_READY = recorder.is_ready()
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
                current_move_image = move_image_img_array.copy()
                
                # wrist image
                if WRIST:
                    current_wrist_image = wrist_img_array.copy()
                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    # move image recorde
                    if args.move_image:
                        left_raw  = current_move_image[:, :move_image_img_shape[1]//2]
                        right_raw = current_move_image[:, move_image_img_shape[1]//2:]

                        # Resize to 480 x 640 each
                        left_up  = cv2.resize(left_raw,  (640, 480))  # (width=640, height=480)
                        right_up = cv2.resize(right_raw, (640, 480))

                        colors["color_4"] = left_up
                        colors["color_5"] = right_up
                        #colors["color_6"] = left_raw
                        #colors["color_7"] = right_raw
                    if BINOCULAR:
                        colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                        colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                        if WRIST:
                            colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                    else:
                        colors[f"color_{0}"] = current_tv_image
                        if WRIST:
                            colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
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
                        "head_offset": {
                            "qpos": [dx, dy],
                            "qvel": [],
                            "torque": []
                        },
                        
                        "controller": {
                            "qpos": controller_combination, #controller_combination if isinstance(controller_combination, list) else controller_combination.tolist(), 
                            "qvel": [],
                            "torque": [],
                        },
                        
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

                        "head_offset": {
                            "qpos": [delta_dx, delta_dy],   # camera “look” action
                            "qvel": [],
                            "torque": [],
                        },
                        "head_offset_original": {
                            "qpos": [dx, dy],   # camera “look” action
                            "qvel": [],
                            "torque": [],
                        },
                        
                        # "odometry": {
                        #     "qpos": [], #keep empty for now
                        #     "qvel": [],
                        #     "torque": [],
                        # },
                        "controller": {
                            "qpos": controller_combination, #controller_combination if isinstance(controller_combination, list) else controller_combination.tolist(), 
                            "qvel": [],
                            "torque": [],
                        },
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

    # except KeyboardInterrupt:
    #     logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logger_mp.error(f"Exception in main loop: {e}", exc_info=True)
        STOP = True
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()

        if args.ipc:
            ipc_server.stop()
        else:
            stop_listening()
            listen_keyboard_thread.join()

        if args.sim:
            sim_state_subscriber.stop_subscribe()
        tv_img_shm.close()
        tv_img_shm.unlink()
        if WRIST:
            wrist_img_shm.close()
            wrist_img_shm.unlink()
        if args.move_image:
            # Use active camera for VR headset with full resolution
            move_image_img_shm.close()
            move_image_img_shm.unlink()

        if args.record:
            recorder.close()
        logger_mp.info("Finally, exiting program.")
        exit(0)