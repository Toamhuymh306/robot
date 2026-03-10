"""
Module 9: Robot Control
Controls 3-DOF robot arm with inverse kinematics and servo control
"""

import numpy as np
import math
from typing import Tuple, Optional, List
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RobotController:
    """
    Robot arm controller for 3-DOF arm.
    Handles inverse kinematics, servo control, and serial communication.
    
    Arm configuration (all lengths in mm):
        - Link 1 (L1): Base to shoulder (vertical rotation)
        - Link 2 (L2): Shoulder to elbow
        - Link 3 (L3): Elbow to end effector
    
    Servo assignments:
        - Servo 1: Base rotation (θ1)
        - Servo 2: Shoulder angle (θ2)
        - Servo 3: Elbow angle (θ3)
    """
    
    def __init__(self,
                 L1: float = config.LINK_1_LENGTH,
                 L2: float = config.LINK_2_LENGTH,
                 L3: float = config.LINK_3_LENGTH,
                 simulation: bool = config.SIMULATION_MODE):
        """
        Initialize robot controller.
        
        Args:
            L1: Link 1 length (base height) in mm
            L2: Link 2 length (upper arm) in mm
            L3: Link 3 length (forearm) in mm
            simulation: If True, simulate robot without real hardware
        """
        # Link lengths
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        
        # Simulation mode
        self.simulation = simulation
        
        # Current joint angles (degrees)
        self.current_angles = list(config.HOME_POSITION)
        
        # Serial connection
        self.serial = None
        self.is_connected = False
        
        # Servo limits
        self.servo_limits = [
            (config.SERVO_1_MIN, config.SERVO_1_MAX),
            (config.SERVO_2_MIN, config.SERVO_2_MAX),
            (config.SERVO_3_MIN, config.SERVO_3_MAX),
        ]
        
        # Movement speed (degrees per step)
        self.speed = 2
        
        # Gripper state
        self.gripper_closed = False
    
    def connect(self, port: str = config.SERIAL_PORT, 
                baudrate: int = config.SERIAL_BAUDRATE) -> bool:
        """
        Connect to Arduino via serial port.
        
        Args:
            port: Serial port name
            baudrate: Communication baud rate
            
        Returns:
            True if connection successful
        """
        if self.simulation:
            print("[SIM] Running in simulation mode - no real connection")
            self.is_connected = True
            return True
        
        try:
            import serial
            self.serial = serial.Serial(port, baudrate, timeout=config.SERIAL_TIMEOUT)
            time.sleep(2)  # Wait for Arduino reset
            self.is_connected = True
            print(f"[INFO] Connected to robot on {port}")
            return True
        except ImportError:
            print("[WARNING] pyserial not installed. Running in simulation mode.")
            self.simulation = True
            self.is_connected = True
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            self.simulation = True  # Fallback to simulation
            self.is_connected = True
            return True
    
    def disconnect(self):
        """Disconnect from robot."""
        if self.serial is not None:
            self.serial.close()
            self.serial = None
        self.is_connected = False
        print("[INFO] Disconnected from robot")
    
    def inverse_kinematics(self, x: float, y: float, z: float) -> Optional[Tuple[float, float, float]]:
        """
        Calculate joint angles for target position using inverse kinematics.
        
        Args:
            x: Target X position (mm)
            y: Target Y position (mm)
            z: Target Z position (mm)
            
        Returns:
            Tuple of (θ1, θ2, θ3) in degrees, None if unreachable
        """
        # Base rotation angle (θ1)
        theta1 = math.atan2(y, x)
        
        # Project to 2D plane (r-z plane)
        r = math.sqrt(x**2 + y**2)  # Horizontal distance
        z_adj = z - self.L1  # Adjust for base height
        
        # Distance from shoulder to target
        d = math.sqrt(r**2 + z_adj**2)
        
        # Check if target is reachable
        if d > self.L2 + self.L3:
            print(f"[WARNING] Target ({x}, {y}, {z}) is too far: d={d:.1f}, max={self.L2 + self.L3}")
            return None
        if d < abs(self.L2 - self.L3):
            print(f"[WARNING] Target ({x}, {y}, {z}) is too close: d={d:.1f}")
            return None
        
        # Elbow angle (θ3) using cosine law
        cos_theta3 = (d**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_theta3 = np.clip(cos_theta3, -1, 1)  # Clamp for numerical stability
        theta3 = math.acos(cos_theta3)
        
        # Shoulder angle (θ2)
        alpha = math.atan2(z_adj, r)  # Angle to target
        cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
        cos_beta = np.clip(cos_beta, -1, 1)
        beta = math.acos(cos_beta)  # Angle in triangle
        
        theta2 = alpha + beta
        
        # Convert to degrees
        theta1_deg = math.degrees(theta1)
        theta2_deg = math.degrees(theta2)
        theta3_deg = math.degrees(theta3)
        
        # Adjust for servo orientation (may need calibration)
        # Standard servo: 0° at one extreme, 180° at other
        theta1_servo = 90 + theta1_deg  # Center at 90°
        theta2_servo = 90 + theta2_deg
        theta3_servo = 180 - theta3_deg  # Inverted for elbow
        
        return (theta1_servo, theta2_servo, theta3_servo)
    
    def forward_kinematics(self, theta1: float, theta2: float, theta3: float) -> Tuple[float, float, float]:
        """
        Calculate end effector position from joint angles.
        
        Args:
            theta1: Base angle (degrees)
            theta2: Shoulder angle (degrees)
            theta3: Elbow angle (degrees)
            
        Returns:
            Tuple of (x, y, z) position in mm
        """
        # Convert to radians
        t1 = math.radians(theta1 - 90)  # Remove offset
        t2 = math.radians(theta2 - 90)
        t3 = math.radians(180 - theta3)
        
        # Calculate position
        r = self.L2 * math.cos(t2) + self.L3 * math.cos(t2 + t3)
        z = self.L1 + self.L2 * math.sin(t2) + self.L3 * math.sin(t2 + t3)
        
        x = r * math.cos(t1)
        y = r * math.sin(t1)
        
        return (x, y, z)
    
    def move_to_angles(self, angles: Tuple[float, float, float], 
                       smooth: bool = True) -> bool:
        """
        Move to specified joint angles.
        
        Args:
            angles: Target angles (θ1, θ2, θ3) in degrees
            smooth: If True, use smooth interpolated movement
            
        Returns:
            True if movement successful
        """
        # Validate angles
        target_angles = []
        for i, (angle, (min_a, max_a)) in enumerate(zip(angles, self.servo_limits)):
            clamped = np.clip(angle, min_a, max_a)
            if clamped != angle:
                print(f"[WARNING] Angle {i} clamped from {angle:.1f} to {clamped:.1f}")
            target_angles.append(clamped)
        
        if smooth:
            return self._smooth_move(target_angles)
        else:
            return self._direct_move(target_angles)
    
    def _smooth_move(self, target_angles: List[float]) -> bool:
        """Smooth movement with interpolation."""
        start_angles = self.current_angles.copy()
        
        # Calculate number of steps
        max_diff = max(abs(t - s) for t, s in zip(target_angles, start_angles))
        num_steps = max(1, int(max_diff / self.speed))
        
        for step in range(1, num_steps + 1):
            t = step / num_steps
            intermediate = [s + t * (e - s) for s, e in zip(start_angles, target_angles)]
            self._direct_move(intermediate)
            
            if self.simulation:
                time.sleep(config.SIMULATION_DELAY / num_steps)
        
        return True
    
    def _direct_move(self, angles: List[float]) -> bool:
        """Direct movement to angles."""
        self.current_angles = list(angles)
        
        if self.simulation:
            self._print_simulation_status()
            return True
        
        # Send to Arduino
        if self.serial is not None and self.serial.is_open:
            command = f"M{int(angles[0])},{int(angles[1])},{int(angles[2])}\n"
            self.serial.write(command.encode())
            time.sleep(0.02)
        
        return True
    
    def move_to_position(self, x: float, y: float, z: float) -> bool:
        """
        Move end effector to Cartesian position.
        
        Args:
            x, y, z: Target position in mm
            
        Returns:
            True if movement successful
        """
        angles = self.inverse_kinematics(x, y, z)
        
        if angles is None:
            print(f"[ERROR] Cannot reach position ({x}, {y}, {z})")
            return False
        
        return self.move_to_angles(angles)
    
    def move_to_cell(self, row: int, col: int, 
                     height_offset: float = 0) -> bool:
        """
        Move to a board cell position.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            height_offset: Additional height offset (mm)
            
        Returns:
            True if movement successful
        """
        from modules.coordinate_mapping import CoordinateMapper
        
        mapper = CoordinateMapper()
        coords = mapper.index_to_coords(row, col)
        
        if coords is None:
            return False
        
        coords[2] += height_offset
        return self.move_to_position(coords[0], coords[1], coords[2])
    
    def grip_close(self):
        """Close gripper."""
        self.gripper_closed = True
        
        if self.simulation:
            print("[SIM] Gripper CLOSED")
            return
        
        if self.serial is not None:
            self.serial.write(b"G1\n")
            time.sleep(0.5)
    
    def grip_open(self):
        """Open gripper."""
        self.gripper_closed = False
        
        if self.simulation:
            print("[SIM] Gripper OPEN")
            return
        
        if self.serial is not None:
            self.serial.write(b"G0\n")
            time.sleep(0.5)
    
    def go_home(self):
        """Move to home position."""
        print("[INFO] Moving to home position...")
        self.move_to_angles(config.HOME_POSITION)
        self.grip_open()
    
    def pick_and_place(self, target_row: int, target_col: int,
                       piece_idx: int = 0) -> bool:
        """
        Execute pick and place operation.
        
        Args:
            target_row: Target row on board
            target_col: Target column on board
            piece_idx: Index of piece in storage
            
        Returns:
            True if operation successful
        """
        from modules.coordinate_mapping import CoordinateMapper
        
        mapper = CoordinateMapper()
        trajectory = mapper.get_pick_and_place_trajectory(
            target_row, target_col, piece_idx
        )
        
        print(f"\n[INFO] Pick and place: piece {piece_idx} -> cell ({target_row}, {target_col})")
        
        for i, waypoint in enumerate(trajectory):
            pos = waypoint['position']
            action = waypoint['action']
            
            print(f"  Step {i+1}: {action} -> ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            
            if action == 'move':
                success = self.move_to_position(pos[0], pos[1], pos[2])
                if not success:
                    print(f"[ERROR] Failed to reach position")
                    return False
            elif action == 'grip_close':
                self.grip_close()
            elif action == 'grip_open':
                self.grip_open()
            
            time.sleep(0.1)
        
        print("[INFO] Pick and place completed")
        return True
    
    def _print_simulation_status(self):
        """Print current simulation status."""
        pos = self.forward_kinematics(*self.current_angles)
        print(f"[SIM] Angles: ({self.current_angles[0]:.1f}°, "
              f"{self.current_angles[1]:.1f}°, {self.current_angles[2]:.1f}°) "
              f"-> Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm")
    
    def get_status(self) -> dict:
        """Get current robot status."""
        pos = self.forward_kinematics(*self.current_angles)
        return {
            'connected': self.is_connected,
            'simulation': self.simulation,
            'angles': tuple(self.current_angles),
            'position': pos,
            'gripper_closed': self.gripper_closed
        }


def test_robot_control():
    """
    Test function for robot control module.
    """
    print("=" * 50)
    print("Testing Robot Control Module")
    print("=" * 50)
    
    robot = RobotController(simulation=True)
    robot.connect()
    
    # Test inverse kinematics
    print("\n[TEST] Inverse Kinematics:")
    test_positions = [
        (100, 0, 50),
        (100, 50, 30),
        (100, -50, 30),
        (150, 0, 0),
    ]
    
    for pos in test_positions:
        angles = robot.inverse_kinematics(*pos)
        if angles:
            fk_pos = robot.forward_kinematics(*angles)
            print(f"  Target: {pos}")
            print(f"  Angles: ({angles[0]:.1f}°, {angles[1]:.1f}°, {angles[2]:.1f}°)")
            print(f"  FK Check: ({fk_pos[0]:.1f}, {fk_pos[1]:.1f}, {fk_pos[2]:.1f})")
        else:
            print(f"  Target: {pos} - UNREACHABLE")
    
    # Test movement
    print("\n[TEST] Movement:")
    robot.go_home()
    
    print("\n[TEST] Moving to position (100, 0, 50):")
    robot.move_to_position(100, 0, 50)
    
    # Test pick and place
    print("\n[TEST] Pick and Place simulation:")
    robot.pick_and_place(1, 1, piece_idx=0)
    
    # Get status
    print("\n[TEST] Robot Status:")
    status = robot.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    robot.disconnect()
    print("\n[INFO] Robot control test completed")


if __name__ == "__main__":
    test_robot_control()
