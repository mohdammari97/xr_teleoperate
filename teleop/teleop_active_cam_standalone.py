#!/usr/bin/env python3
"""
Active Camera Control using VR Head Tracking with Threaded ActiveCameraController
This version uses the new threaded camera controller for simplified operation.
"""

import numpy as np
import logging
import os
import sys
import time
import cv2
import argparse
from multiprocessing import shared_memory
from threading import Thread
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.image_server.image_client import ImageClient
from teleop.robot_control.active_head_cam import ActiveCameraController

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
            logging.FileHandler(os.path.join(log_dir, "teleop_active_cam_standalone.log")),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    # Only show warnings and errors on console by default
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING if not verbose else logging.DEBUG)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Replace the console handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger('teleop_active_cam_standalone')

def main():
    parser = argparse.ArgumentParser(description="Active Camera Control using VR Head Tracking - Threaded Version")
    parser.add_argument('--port', type=str, default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3R4A5A-if00-port0", 
                       help="Serial port for the Dynamixel servo controller")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--safe-mode', action='store_true', default=False, help='Enable safe mode with limited movement (default: True)')
    parser.add_argument('--max-movement', type=float, default=60.0, help='Maximum movement in degrees from start position (default: 30.0° for safety)')
    parser.add_argument('--use-opencv', action='store_true', default=True, help='Use OpenCV camera instead of ZED')
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info(f"Starting active camera control with DynamixelAgent on port: {args.port}")

    # Image configuration for OpenCV camera
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 640],  # Single camera resolution
        'head_camera_id_numbers': [0],  # Use camera 0
    }
    
    # Calculate image shape and create shared memory
    tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    # Initialize image client
    img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name)
    image_receive_thread = Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()
    logger.info("Image receive thread started")

    # Initialize TeleVision wrapper (for VR head tracking)
    tv_wrapper = TeleVisionWrapper(False, tv_img_shape, tv_img_shm.name, ngrok=True)  # False = monocular
    logger.info("TeleVision wrapper initialized")

    # Initialize camera controller
    camera_controller = None
    
    try:
        # Initialize camera controller with threading support
        logger.info("Initializing threaded camera controller...")
        camera_controller = ActiveCameraController(
            port=args.port,
            safe_mode=args.safe_mode,
            max_movement_deg=args.max_movement,
            logger=logger
        )
        
        # Connect to servos
        logger.info("Connecting to camera servos...")
        if not camera_controller.connect():
            logger.error("Failed to connect to servos")
            return

        # Get initial positions for reference
        start_pitch_rad, start_yaw_rad = camera_controller.start_positions
        logger.info(f"Initial servo positions (degrees): Pitch={np.rad2deg(start_pitch_rad):.2f}, Yaw={np.rad2deg(start_yaw_rad):.2f}")
        
        # Wait for user input in terminal to start teleoperation
        print("\n" + "="*60)
        print("ACTIVE CAMERA TELEOPERATION READY")
        print("="*60)
        print(f"Safe Mode: {'ENABLED' if args.safe_mode else 'DISABLED'}")
        print(f"Max Movement: ±{args.max_movement:.1f}° from start position")
        print("Press 'r' and ENTER to start teleoperation, or 'q' and ENTER to quit:")
        
        while True:
            user_input = input().strip().lower()
            if user_input == 'r':
                logger.info("Starting teleoperation...")
                break
            elif user_input == 'q':
                logger.info("Shutdown requested by user")
                raise KeyboardInterrupt("Shutdown requested by user")
            else:
                print("Invalid input. Press 'r' to start or 'q' to quit:")

        # Enable head tracking (this starts the background thread)
        logger.info("Enabling head tracking...")
        if not camera_controller.enable_head_tracking(tv_wrapper):
            logger.error("Failed to enable head tracking")
            return
        
        logger.info("Head tracking enabled! Camera will now follow your head movements automatically.")
        logger.info("Press 'q' in the camera window or Ctrl+C to stop...")

        # Main display loop - much simpler now that servo control is threaded
        frame_counter = 0
        while True:
            # Get servo states from threaded controller
            servo_states = camera_controller.get_servo_states()
            
            # Display the camera feed with overlay information
            frame = tv_img_array.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0)  # Green
            thickness = 2
            
            # Display current servo positions
            current_text = f"Current: Pitch={np.rad2deg(servo_states['current_pitch']):.1f}° Yaw={np.rad2deg(servo_states['current_yaw']):.1f}°"
            cv2.putText(frame, current_text, (10, 30), font, font_scale, color, thickness)
            
            # Display target servo positions
            target_text = f"Target: Pitch={np.rad2deg(servo_states['target_pitch']):.1f}° Yaw={np.rad2deg(servo_states['target_yaw']):.1f}°"
            cv2.putText(frame, target_text, (10, 60), font, font_scale, (0, 255, 255), thickness)
            
            # Display movement limits and mode
            mode_text = f"Mode: {'SAFE' if args.safe_mode else 'NORMAL'} | Limits: ±{args.max_movement:.1f}°"
            cv2.putText(frame, mode_text, (10, 90), font, font_scale, (255, 255, 0), thickness)
            
            # Display status
            status_text = "HEAD TRACKING ACTIVE - Move your head to control camera"
            cv2.putText(frame, status_text, (10, 120), font, font_scale, (255, 0, 255), thickness)
            
            cv2.imshow('Active Camera Teleop', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("'q' pressed, shutting down.")
                break
            
            # Log servo positions periodically
            if frame_counter % 150 == 0:  # Every 5 seconds at 30fps
                logger.info(f"Servo Status - Current: P={np.rad2deg(servo_states['current_pitch']):.1f}° Y={np.rad2deg(servo_states['current_yaw']):.1f}° | "
                           f"Target: P={np.rad2deg(servo_states['target_pitch']):.1f}° Y={np.rad2deg(servo_states['target_yaw']):.1f}°")
            
            frame_counter += 1
            time.sleep(1/30.0)  # 30fps display loop

    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    
    finally:
        # Cleanup
        logger.info("Shutting down...")
        
        if camera_controller:
            # This will stop the head tracking thread and disconnect from servos
            camera_controller.disconnect()
            logger.info("Camera controller disconnected and threads stopped")
        
        cv2.destroyAllWindows()
        
        try:
            tv_img_shm.unlink()
            tv_img_shm.close()
            logger.info("Shared memory cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up shared memory: {e}")
        
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()
