#!/usr/bin/env python3
"""
Active Camera Control using dynamixel servos.
"""

import numpy as np
import logging
import os
import sys
import time
import threading
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.dynamixel.active_cam import DynamixelAgent
from teleop.robot_control.dynamixel.error_analyzer import DynamixelErrorAnalyzer


class ActiveCameraController:
    """Controller for the active camera servo system using DynamixelAgent with threaded head tracking."""
    
    def __init__(self, port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3R4A5A-if00-port0", 
                 pitch_id=1, yaw_id=2, safe_mode=False, max_movement_deg=60.0, 
                 tv_wrapper=None, logger=None):
        """Initialize the camera controller with two servos for pitch and yaw.
        
        Args:
            port: Serial port for the U2D2 interface
            pitch_id: Dynamixel ID for pitch servo (vertical movement)  
            yaw_id: Dynamixel ID for yaw servo (horizontal movement)
            safe_mode: Enable safe mode with limited movement (0.1x scaling)
            max_movement_deg: Maximum movement in degrees from start position
            tv_wrapper: TeleVisionWrapper instance for head tracking (optional)
            logger: Logger instance (optional)
        """
        self.port = port
        self.pitch_id = pitch_id
        self.yaw_id = yaw_id
        self.safe_mode = safe_mode
        self.max_movement_rad = np.radians(max_movement_deg)
        self.tv_wrapper = tv_wrapper
        self.logger = logger or logging.getLogger('ActiveCameraController')
        
        # Starting positions in radians (safe positions)
        self.start_positions = np.array([195.0 * np.pi / 180, 90.0 * np.pi / 180])
        
        # Initialize DynamixelAgent
        self.agent = None
        self.connected = False
        
        # Initialize error analyzer
        self.error_analyzer = DynamixelErrorAnalyzer(port=self.port)
        
        # Threading control for head tracking
        self.running = False
        self.thread = None
        
        # Shared state for recording (thread-safe)
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.state_lock = threading.Lock()
        
        # Head tracking reference
        self.initial_head_rotation = None
        self.head_tracking_enabled = False
    
    def analyze_servo_errors(self):
        """Analyze servo errors using the error analyzer."""
        try:
            print("\n" + "="*60)
            print("RUNNING SERVO ERROR ANALYSIS...")
            print("="*60)
            
            analysis = self.error_analyzer.analyze_system([self.pitch_id, self.yaw_id])
            self.error_analyzer.print_analysis_report(analysis)
            
            return analysis
        except Exception as e:
            logging.error(f"Error during servo analysis: {e}")
            return None

    def connect(self):
        """Connect to servos using DynamixelAgent."""
        try:
            # First, run error analysis to check servo health
            print("Checking servo health before connection...")
            servo_analysis = self.analyze_servo_errors()
            
            # Check if servos are responding
            if servo_analysis:
                failed_servos = []
                for servo_id, servo_data in servo_analysis.get('servos', {}).items():
                    if not servo_data.get('communication_ok', False):
                        failed_servos.append(servo_id)
                
                if failed_servos:
                    error_msg = f"Servos {failed_servos} are not responding. Check connections and power."
                    logging.error(error_msg)
                    print(f"\n❌ ERROR: {error_msg}")
                    print("Please check:")
                    print("  1. Servo power supply (12V)")
                    print("  2. USB cable connections")
                    print("  3. U2D2 interface")
                    print("  4. Run reduce_latency.sh script")
                    raise RuntimeError(error_msg)
                else:
                    print("✅ All servos responding normally")
            
            self.agent = DynamixelAgent(port=self.port, start_joints=self.start_positions)
            
            # Enable torque
            print("Enabling servo torque...")
            self.agent._robot.set_torque_mode(True)
            
            # Now that we're connected, mark as connected
            self.connected = True
            logging.info(f"Connected to active camera servos on {self.port}")
            
            # Perform safety check on initial positions
            self.check_initial_positions(tolerance_deg=20.0)
            
            return True
        except Exception as e:
            logging.error(f"Failed to connect to servos: {e}")
            self.disconnect()
            raise
    
    def disconnect(self):
        """Disconnect from servos."""
        try:
            # Stop head tracking thread first
            self.stop_head_tracking()
            
            if self.agent and hasattr(self.agent, '_robot'):
                self.agent._robot.set_torque_mode(False)
                self.logger.info("Disabled servo torque and disconnected")
        except Exception as e:
            self.logger.warning(f"Error during disconnect: {e}")
        finally:
            self.connected = False
    
    def enable_head_tracking(self, tv_wrapper):
        """Enable head tracking with a TeleVision wrapper.
        
        Args:
            tv_wrapper: TeleVisionWrapper instance for head tracking
        """
        if not self.connected:
            raise RuntimeError("Must connect to servos before enabling head tracking")
        
        self.tv_wrapper = tv_wrapper
        
        # Wait for head tracking data and capture initial position
        self.logger.info("Waiting for initial head tracking data...")
        timeout = 10.0  # 10 second timeout
        start_time = time.time()
        
        while self.tv_wrapper.get_head_orientation() is None:
            if time.time() - start_time > timeout:
                self.logger.error("Timeout waiting for head tracking data")
                return False
            time.sleep(0.1)
        
        # Capture initial head rotation as reference
        self.initial_head_rotation = R.from_quat(self.tv_wrapper.get_head_orientation())
        
        # Initialize shared state with current positions
        with self.state_lock:
            self.current_pitch = self.start_positions[0]
            self.current_yaw = self.start_positions[1]
            self.target_pitch = self.start_positions[0]
            self.target_yaw = self.start_positions[1]
        
        self.logger.info(f"Initial camera positions: Pitch={np.rad2deg(self.start_positions[0]):.2f}°, Yaw={np.rad2deg(self.start_positions[1]):.2f}°")
        
        # Start the control thread
        self.running = True
        self.thread = threading.Thread(target=self._head_tracking_loop, daemon=True)
        self.thread.start()
        self.head_tracking_enabled = True
        self.logger.info("Head tracking started")
        return True
    
    def stop_head_tracking(self):
        """Stop the head tracking thread."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)
                self.thread = None
            self.head_tracking_enabled = False
            self.logger.info("Head tracking stopped")
    
    def get_servo_states(self):
        """Get current servo states for recording (thread-safe).
        
        Returns:
            dict: Dictionary with current and target positions in radians
        """
        with self.state_lock:
            return {
                'current_pitch': self.current_pitch,
                'current_yaw': self.current_yaw,
                'target_pitch': self.target_pitch,
                'target_yaw': self.target_yaw
            }
    
    def _head_tracking_loop(self):
        """Main head tracking control loop running in separate thread."""
        try:
            while self.running:
                if not (self.connected and self.tv_wrapper):
                    time.sleep(0.02)
                    continue
                
                # Get current head rotation
                current_head_quat = self.tv_wrapper.get_head_orientation()
                if current_head_quat is None:
                    time.sleep(0.01)
                    continue
                
                current_head_rotation = R.from_quat(current_head_quat)
                
                # Calculate relative rotation from initial position
                relative_rotation = current_head_rotation * self.initial_head_rotation.inv()
                euler_angles = relative_rotation.as_euler('xyz', degrees=True)
                
                # Extract pitch and yaw changes
                pitch_delta_deg = euler_angles[1]  # Rotation around Y-axis
                yaw_delta_deg = -euler_angles[2]   # Rotation around Z-axis (inverted)
                
                # Apply scaling factor
                scaling_factor = 0.1 if self.safe_mode else 1.0
                pitch_movement_deg = pitch_delta_deg * scaling_factor
                yaw_movement_deg = yaw_delta_deg * scaling_factor
                
                # Calculate target positions
                target_pitch_rad = self.start_positions[0] + np.deg2rad(pitch_movement_deg)
                target_yaw_rad = self.start_positions[1] - np.deg2rad(yaw_movement_deg)
                
                # Apply safety limits
                min_pitch = self.start_positions[0] - self.max_movement_rad
                max_pitch = self.start_positions[0] + self.max_movement_rad
                min_yaw = self.start_positions[1] - self.max_movement_rad
                max_yaw = self.start_positions[1] + self.max_movement_rad
                
                target_pitch_rad = np.clip(target_pitch_rad, min_pitch, max_pitch)
                target_yaw_rad = np.clip(target_yaw_rad, min_yaw, max_yaw)
                
                # Send commands to servos
                try:
                    self.set_positions(target_pitch_rad, target_yaw_rad)
                    
                    # Update shared state
                    current_positions = self.get_positions()
                    with self.state_lock:
                        self.current_pitch = current_positions[0]
                        self.current_yaw = current_positions[1]
                        self.target_pitch = target_pitch_rad
                        self.target_yaw = target_yaw_rad
                        
                except Exception as e:
                    self.logger.warning(f"Head tracking control error: {e}")
                
                time.sleep(0.02)  # 50Hz control loop
                
        except Exception as e:
            self.logger.error(f"Head tracking thread error: {e}")
        finally:
            self.logger.info("Head tracking thread exiting")
    
    def get_positions(self):
        """Get current positions of both servos in radians.
        
        Returns:
            tuple: (pitch_position, yaw_position) in radians
        """
        if not self.connected:
            raise RuntimeError("Not connected to servos")
        
        try:
            # Get joint state returns [pitch, yaw] in radians
            positions = self.agent.act({})
            return tuple(positions)
        except Exception as e:
            logging.error(f"Error reading servo positions: {e}")
            raise # Re-raise the exception to be handled by the caller
    
    def set_positions(self, pitch_rad, yaw_rad):
        """Set target positions for both servos.
        
        Args:
            pitch_rad: Target pitch position in radians
            yaw_rad: Target yaw position in radians
        """
        if not self.connected:
            raise RuntimeError("Not connected to servos")
        
        try:
            # Command joint state with [pitch, yaw] in radians
            target_positions = [pitch_rad, yaw_rad]
            
            # Debug: Check if target positions are reasonable
            target_deg = [np.rad2deg(pitch_rad), np.rad2deg(yaw_rad)]
            if target_deg[0] < 0 or target_deg[0] > 360 or target_deg[1] < 0 or target_deg[1] > 360:
                logging.warning(f"Target positions outside typical servo range: Pitch={target_deg[0]:.1f}°, Yaw={target_deg[1]:.1f}°")
            
            self.agent._robot.command_joint_state(target_positions)
        except Exception as e:
            logging.error(f"Error setting servo positions: {e}")
            logging.error(f"Target positions: Pitch={np.rad2deg(pitch_rad):.2f}°, Yaw={np.rad2deg(yaw_rad):.2f}°")
    
    def check_initial_positions(self, tolerance_deg=20.0):
        """
        Check if the initial positions of the servos are within a safe tolerance.
        If not, raises a RuntimeError to stop the script.
        """
        if not self.connected:
            raise RuntimeError("Cannot check initial positions, not connected to servos.")

        logging.info("Checking initial servo positions for safety...")
        
        try:
            current_positions_rad = self.get_positions()
            current_positions_deg = np.rad2deg(current_positions_rad)
            start_positions_deg = np.rad2deg(self.start_positions)

            pitch_diff = abs(current_positions_deg[0] - start_positions_deg[0])
            yaw_diff = abs(current_positions_deg[1] - start_positions_deg[1])

            if pitch_diff > tolerance_deg or yaw_diff > tolerance_deg:
                error_msg = (
                    f"\n!!! SAFETY ALERT: SERVO POSITION OUT OF TOLERANCE !!!\n"
                    f"Initial servo position deviates by more than {tolerance_deg}° from the expected start.\n"
                    f"------------------------------------------------------------------------------------\n"
                    f"Pitch Servo (ID {self.pitch_id}):\n"
                    f"  - Current Position: {current_positions_deg[0]:.2f}°\n"
                    f"  - Expected Start:   {start_positions_deg[0]:.2f}°\n"
                    f"  - Deviation:        {pitch_diff:.2f}°\n"
                    f"Yaw Servo (ID {self.yaw_id}):\n"
                    f"  - Current Position: {current_positions_deg[1]:.2f}°\n"
                    f"  - Expected Start:   {start_positions_deg[1]:.2f}°\n"
                    f"  - Deviation:        {yaw_diff:.2f}°\n"
                    f"------------------------------------------------------------------------------------\n"
                    f"This may indicate a physical obstruction or a desynchronization. "
                    f"Please check the hardware before restarting.\n"
                )
                logging.error(error_msg)
                raise RuntimeError("Initial servo position check failed. Aborting for safety.")
            
            logging.info("Initial servo positions are within safe limits. Continuing.")

        except Exception as e:
            logging.error(f"Failed to perform initial position check: {e}")
            raise

    def is_moving(self):
        """Check if either servo is currently moving.
        
        Returns:
            bool: True if any servo is moving (placeholder for now)
        """
        if not self.connected:
            return False
        
        # For now, assume movement check is not available in DynamixelAgent
        # Could be implemented by checking position changes over time
        return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()