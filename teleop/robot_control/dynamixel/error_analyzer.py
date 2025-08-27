#!/usr/bin/env python3
"""
Comprehensive Dynamixel Error Analyzer Module
Provides detailed error diagnosis for Dynamixel servos including hardware, communication, and operational errors.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import dynamixel_sdk
from dynamixel_sdk import (
    PortHandler, 
    PacketHandler, 
    COMM_SUCCESS,
    COMM_PORT_BUSY,
    COMM_TX_FAIL,
    COMM_RX_FAIL, 
    COMM_TX_ERROR,
    COMM_RX_WAITING,
    COMM_RX_TIMEOUT,
    COMM_RX_CORRUPT,
    COMM_NOT_AVAILABLE
)

class DynamixelErrorAnalyzer:
    """Comprehensive error analyzer for Dynamixel servos."""
    
    # Control table addresses for XM430-W350-T and similar models
    ADDR_TORQUE_ENABLE = 64
    ADDR_PRESENT_POSITION = 132
    ADDR_PRESENT_VELOCITY = 128
    ADDR_PRESENT_CURRENT = 126
    ADDR_PRESENT_TEMPERATURE = 146
    ADDR_PRESENT_VOLTAGE = 144
    ADDR_HARDWARE_ERROR_STATUS = 70
    ADDR_SHUTDOWN = 63
    
    # Hardware error bit flags
    HARDWARE_ERROR_FLAGS = {
        0: "Input Voltage Error",
        1: "Motor Hall Sensor Error", 
        2: "Overheating Error",
        3: "Motor Encoder Error",
        4: "Electrical Shock Error",
        5: "Overload Error"
    }
    
    # Communication error codes
    COMM_ERROR_CODES = {
        COMM_SUCCESS: "Success",
        COMM_PORT_BUSY: "Port is busy",
        COMM_TX_FAIL: "Failed to transmit instruction packet",
        COMM_RX_FAIL: "Failed to receive status packet",
        COMM_TX_ERROR: "Incorrect instruction packet",
        COMM_RX_WAITING: "Waiting for status packet",
        COMM_RX_TIMEOUT: "Status packet timeout",
        COMM_RX_CORRUPT: "Status packet corrupt",
        COMM_NOT_AVAILABLE: "Protocol does not support this function"
    }
    
    # Protocol error flags
    PROTOCOL_ERROR_FLAGS = {
        0x01: "Result Fail",
        0x02: "Instruction Error", 
        0x03: "CRC Error",
        0x04: "Data Range Error",
        0x05: "Data Length Error",
        0x06: "Data Limit Error",
        0x07: "Access Error"
    }
    
    def __init__(self, port: str = None, port_name: str = None, baudrate: int = 2000000, protocol_version: float = 2.0):
        """Initialize the error analyzer.
        
        Args:
            port: Serial port name (e.g., '/dev/ttyUSB0') - alternative parameter name
            port_name: Serial port name (e.g., '/dev/ttyUSB0')
            baudrate: Communication baudrate
            protocol_version: Dynamixel protocol version
        """
        self.port_name = port if port is not None else port_name
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        
        # Initialize PortHandler and PacketHandler
        self.port_handler = PortHandler(self.port_name)
        self.packet_handler = PacketHandler(protocol_version)
        
        self.logger = logging.getLogger('DynamixelErrorAnalyzer')
        
        # Store last known states
        self.last_positions = {}
        self.last_temperatures = {}
        self.last_voltages = {}
        
    def connect(self) -> bool:
        """Connect to the Dynamixel port.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.port_handler.openPort():
                self.logger.info(f"Successfully opened port {self.port_name}")
                
                if self.port_handler.setBaudRate(self.baudrate):
                    self.logger.info(f"Successfully set baudrate to {self.baudrate}")
                    return True
                else:
                    self.logger.error(f"Failed to set baudrate to {self.baudrate}")
                    return False
            else:
                self.logger.error(f"Failed to open port {self.port_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception during connection: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the Dynamixel port."""
        try:
            self.port_handler.closePort()
            self.logger.info("Port closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing port: {e}")
    
    def ping_servo(self, servo_id: int) -> Tuple[bool, str]:
        """Ping a servo to check basic connectivity.
        
        Args:
            servo_id: ID of the servo to ping
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(self.port_handler, servo_id)
            
            if dxl_comm_result != COMM_SUCCESS:
                return False, f"Communication Error: {self.COMM_ERROR_CODES.get(dxl_comm_result, 'Unknown')}"
            elif dxl_error != 0:
                return False, f"Protocol Error: {self._decode_protocol_error(dxl_error)}"
            else:
                return True, f"Servo ID {servo_id} responded (Model: {dxl_model_number})"
                
        except Exception as e:
            return False, f"Exception during ping: {e}"
    
    def read_hardware_errors(self, servo_id: int) -> Dict[str, Any]:
        """Read and decode hardware error status from a servo.
        
        Args:
            servo_id: ID of the servo to check
            
        Returns:
            Dictionary containing error analysis
        """
        result = {
            'servo_id': servo_id,
            'hardware_errors': [],
            'error_byte': None,
            'communication_ok': False,
            'timestamp': time.time()
        }
        
        try:
            # Read hardware error status
            error_byte, dxl_comm_result, dxl_error = self.packet_handler.read1ByteTxRx(
                self.port_handler, servo_id, self.ADDR_HARDWARE_ERROR_STATUS
            )
            
            if dxl_comm_result != COMM_SUCCESS:
                result['communication_error'] = self.COMM_ERROR_CODES.get(dxl_comm_result, 'Unknown')
                return result
            elif dxl_error != 0:
                result['protocol_error'] = self._decode_protocol_error(dxl_error)
                return result
            
            result['communication_ok'] = True
            result['error_byte'] = error_byte
            
            # Decode hardware error flags
            for bit, error_name in self.HARDWARE_ERROR_FLAGS.items():
                if error_byte & (1 << bit):
                    result['hardware_errors'].append(error_name)
            
            return result
            
        except Exception as e:
            result['exception'] = str(e)
            return result
    
    def read_servo_status(self, servo_id: int) -> Dict[str, Any]:
        """Read comprehensive status from a servo.
        
        Args:
            servo_id: ID of the servo to check
            
        Returns:
            Dictionary containing complete servo status
        """
        status = {
            'servo_id': servo_id,
            'timestamp': time.time(),
            'communication_ok': False,
            'errors': []
        }
        
        try:
            # Check basic connectivity first
            ping_ok, ping_msg = self.ping_servo(servo_id)
            if not ping_ok:
                status['errors'].append(f"Ping failed: {ping_msg}")
                return status
            
            status['communication_ok'] = True
            
            # Read various status registers
            readings = {}
            
            # Position
            position, comm_result, error = self.packet_handler.read4ByteTxRx(
                self.port_handler, servo_id, self.ADDR_PRESENT_POSITION
            )
            if comm_result == COMM_SUCCESS and error == 0:
                readings['position'] = position
                readings['position_degrees'] = self._position_to_degrees(position)
            
            # Temperature  
            temp, comm_result, error = self.packet_handler.read1ByteTxRx(
                self.port_handler, servo_id, self.ADDR_PRESENT_TEMPERATURE
            )
            if comm_result == COMM_SUCCESS and error == 0:
                readings['temperature'] = temp
            
            # Voltage
            voltage, comm_result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler, servo_id, self.ADDR_PRESENT_VOLTAGE
            )
            if comm_result == COMM_SUCCESS and error == 0:
                readings['voltage'] = voltage / 10.0  # Convert to volts
            
            # Current
            current, comm_result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler, servo_id, self.ADDR_PRESENT_CURRENT
            )
            if comm_result == COMM_SUCCESS and error == 0:
                readings['current'] = current
            
            # Hardware errors
            hw_errors = self.read_hardware_errors(servo_id)
            readings['hardware_errors'] = hw_errors
            
            status['readings'] = readings
            
            # Analyze for potential issues
            self._analyze_servo_health(status)
            
            return status
            
        except Exception as e:
            status['exception'] = str(e)
            return status
    
    def analyze_system(self, servo_ids: List[int]) -> Dict[str, Any]:
        """Perform comprehensive system analysis.
        
        Args:
            servo_ids: List of servo IDs to analyze
            
        Returns:
            Dictionary containing system-wide analysis
        """
        analysis = {
            'timestamp': time.time(),
            'port': self.port_name,
            'baudrate': self.baudrate,
            'servos': {},
            'system_errors': [],
            'recommendations': []
        }
        
        # Check port connection
        if not self.connect():
            analysis['system_errors'].append("Failed to connect to serial port")
            analysis['recommendations'].append("Check USB connection and port permissions")
            return analysis
        
        try:
            # Analyze each servo
            for servo_id in servo_ids:
                self.logger.info(f"Analyzing servo ID {servo_id}...")
                analysis['servos'][servo_id] = self.read_servo_status(servo_id)
            
            # System-wide analysis
            self._analyze_system_health(analysis)
            
        finally:
            self.disconnect()
        
        return analysis
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Print a formatted analysis report.
        
        Args:
            analysis: Analysis results from analyze_system()
        """
        print("\n" + "="*80)
        print("DYNAMIXEL SYSTEM ANALYSIS REPORT")
        print("="*80)
        print(f"Port: {analysis['port']}")
        print(f"Baudrate: {analysis['baudrate']}")
        print(f"Analysis Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(analysis['timestamp']))}")
        
        # System-level errors
        if analysis['system_errors']:
            print("\n🔴 SYSTEM ERRORS:")
            for error in analysis['system_errors']:
                print(f"  • {error}")
        
        # Servo analysis
        print(f"\n📊 SERVO ANALYSIS ({len(analysis['servos'])} servos):")
        for servo_id, servo_data in analysis['servos'].items():
            self._print_servo_report(servo_id, servo_data)
        
        # Recommendations
        if analysis['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
        
        print("="*80)
    
    def _print_servo_report(self, servo_id: int, servo_data: Dict[str, Any]):
        """Print detailed servo report."""
        print(f"\n  Servo ID {servo_id}:")
        
        if not servo_data.get('communication_ok', False):
            print("    ❌ Communication FAILED")
            if 'errors' in servo_data:
                for error in servo_data['errors']:
                    print(f"       • {error}")
            return
        
        print("    ✅ Communication OK")
        
        if 'readings' in servo_data:
            readings = servo_data['readings']
            
            # Position
            if 'position_degrees' in readings:
                print(f"    📍 Position: {readings['position_degrees']:.1f}° ({readings['position']} raw)")
            
            # Temperature
            if 'temperature' in readings:
                temp = readings['temperature']
                temp_status = "🟢" if temp < 60 else "🟡" if temp < 70 else "🔴"
                print(f"    🌡️  Temperature: {temp}°C {temp_status}")
            
            # Voltage
            if 'voltage' in readings:
                voltage = readings['voltage']
                volt_status = "🟢" if 11.0 <= voltage <= 14.8 else "🟡" if 10.0 <= voltage < 11.0 else "🔴"
                print(f"    ⚡ Voltage: {voltage:.1f}V {volt_status}")
            
            # Current
            if 'current' in readings:
                print(f"    🔌 Current: {readings['current']} mA")
            
            # Hardware errors
            if 'hardware_errors' in readings:
                hw_errors = readings['hardware_errors']
                if hw_errors.get('hardware_errors'):
                    print("    🔴 HARDWARE ERRORS:")
                    for error in hw_errors['hardware_errors']:
                        print(f"       • {error}")
                else:
                    print("    ✅ No hardware errors")
        
        # Health warnings
        if 'warnings' in servo_data:
            for warning in servo_data['warnings']:
                print(f"    ⚠️  {warning}")
    
    def _position_to_degrees(self, position: int) -> float:
        """Convert raw position to degrees."""
        return (position - 2048) * 0.088  # For XM430 series
    
    def _decode_protocol_error(self, error_byte: int) -> str:
        """Decode protocol error byte."""
        if error_byte == 0:
            return "No Error"
        
        errors = []
        for bit, error_name in self.PROTOCOL_ERROR_FLAGS.items():
            if error_byte & bit:
                errors.append(error_name)
        
        return ", ".join(errors) if errors else f"Unknown Error (0x{error_byte:02X})"
    
    def _analyze_servo_health(self, status: Dict[str, Any]):
        """Analyze servo health and add warnings."""
        if 'readings' not in status:
            return
        
        readings = status['readings']
        warnings = []
        
        # Temperature check
        if 'temperature' in readings:
            temp = readings['temperature']
            if temp > 70:
                warnings.append(f"High temperature ({temp}°C) - Risk of thermal shutdown")
            elif temp > 60:
                warnings.append(f"Elevated temperature ({temp}°C) - Monitor closely")
        
        # Voltage check
        if 'voltage' in readings:
            voltage = readings['voltage']
            if voltage < 10.0:
                warnings.append(f"Low voltage ({voltage:.1f}V) - May cause erratic behavior")
            elif voltage > 14.8:
                warnings.append(f"High voltage ({voltage:.1f}V) - Risk of damage")
        
        # Hardware error check
        if 'hardware_errors' in readings:
            hw_errors = readings['hardware_errors']
            if hw_errors.get('hardware_errors'):
                warnings.append("Hardware errors detected - Check servo condition")
        
        if warnings:
            status['warnings'] = warnings
    
    def _analyze_system_health(self, analysis: Dict[str, Any]):
        """Perform system-wide health analysis."""
        servo_count = len(analysis['servos'])
        failed_servos = []
        warning_servos = []
        
        for servo_id, servo_data in analysis['servos'].items():
            if not servo_data.get('communication_ok', False):
                failed_servos.append(servo_id)
            elif 'warnings' in servo_data:
                warning_servos.append(servo_id)
        
        # Add system recommendations
        if failed_servos:
            analysis['recommendations'].append(f"Check connections for servos: {failed_servos}")
            analysis['recommendations'].append("Verify servo IDs and wiring")
            analysis['recommendations'].append("Run reduce_latency.sh if using FTDI USB adapter")
        
        if warning_servos:
            analysis['recommendations'].append(f"Monitor servos with warnings: {warning_servos}")
        
        if not failed_servos and not warning_servos:
            analysis['recommendations'].append("System appears healthy - all servos responding normally")


def main():
    """Main function for standalone testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamixel Error Analyzer")
    parser.add_argument('--port', type=str, 
                       default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3R4A5A-if00-port0",
                       help="Serial port for Dynamixel communication")
    parser.add_argument('--baudrate', type=int, default=2000000, help="Communication baudrate")
    parser.add_argument('--servo-ids', type=int, nargs='+', default=[1, 2], 
                       help="Servo IDs to analyze")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create analyzer and run analysis
    analyzer = DynamixelErrorAnalyzer(args.port, args.baudrate)
    analysis = analyzer.analyze_system(args.servo_ids)
    analyzer.print_analysis_report(analysis)


if __name__ == "__main__":
    main()
