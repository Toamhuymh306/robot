"""
Robot Controller Module
Controls the Tic-Tac-Toe robot arm via serial communication with ESP32.
Supports both real hardware control and simulation mode.
"""

import serial
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


class RobotController:
    """Controls the robot arm for Tic-Tac-Toe game."""
    
    def __init__(self, port: str = None, baud: int = None, simulation_mode: bool = None, *, max_retries: int = 3, retry_delay: float = 1.0, verbose: bool = False):
        self.serial_port = None
        self.connected = False
        # allow caller to override simulation mode/port/baud
        self.simulation_mode = config.SIMULATION_MODE if simulation_mode is None else bool(simulation_mode)
        self.port = config.SERIAL_PORT if port is None else port
        self.baud = config.SERIAL_BAUDRATE if baud is None else int(baud)
        self.connection_error = None
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.verbose = bool(verbose)

        # command retry attempts when no ACK received
        self._command_retry_attempts = 2

        self.command_timeouts = {
            'INIT': 20,
            'HOME': 20,
            'MOVE': 30,
            'PICK': 120,
            'PLACE': 180,
            'DEMO': 90,
            'TEST': 5,
        }
        
        # Board position mapping (0-8) to physical coordinates
        # These should match the positions in the C code
        self.board_positions = [
            (0, 0),  # Position 0: top-left
            (1, 0),  # Position 1: top-center
            (2, 0),  # Position 2: top-right
            (0, 1),  # Position 3: middle-left
            (1, 1),  # Position 4: center
            (2, 1),  # Position 5: middle-right
            (0, 2),  # Position 6: bottom-left
            (1, 2),  # Position 7: bottom-center
            (2, 2),  # Position 8: bottom-right
        ]
        
        # Initialize connection
        self._initialize()
    
    def _initialize(self):
        """Initialize serial connection or simulation mode."""
        if self.simulation_mode:
            print("Robot Controller: SIMULATION MODE (no real robot)")
            self.connected = True
            return

        attempts = 0
        last_exc = None
        while attempts < self.max_retries and not self.connected:
            try:
                print(f"Connecting to robot arm at {self.port} @ {self.baud} (attempt {attempts+1}/{self.max_retries})...")
                self.serial_port = serial.Serial(
                    port=self.port,
                    baudrate=self.baud,
                    timeout=config.SERIAL_TIMEOUT
                )
                time.sleep(2)  # Wait for connection to establish
                self._drain_serial(1.0)
                self.connected = True
                print("Robot arm connected successfully!")

                # Sync with firmware first.
                if not self._send_command("TEST"):
                    print("Warning: TEST handshake failed, continuing with INIT...")

                # Send initialization command
                self._send_command("INIT")
                time.sleep(1)
                break

            except serial.SerialException as e:
                last_exc = e
                self.connection_error = str(e)
                attempts += 1
                backoff = self.retry_delay * (2 ** (attempts - 1))
                print(f"Failed to connect to robot arm: {e} (retry in {backoff:.1f}s)")
                time.sleep(backoff)

        if not self.connected:
            if getattr(config, 'ALLOW_SIMULATION_FALLBACK', False):
                print("Falling back to simulation mode")
                self.simulation_mode = True
                self.connected = True
            else:
                print("REAL mode required. Could not open serial port.")
                if last_exc:
                    print("Last error:", last_exc)
                print("Close idf_monitor/Serial Monitor and retry.")

    def _drain_serial(self, duration_sec=0.5):
        """Drain noisy boot/debug lines from UART so ACK parsing starts clean."""
        if self.simulation_mode or not self.serial_port:
            return

        end_time = time.time() + duration_sec
        try:
            while time.time() < end_time:
                if self.serial_port.in_waiting > 0:
                    self.serial_port.readline()
                else:
                    time.sleep(0.02)
        except Exception:
            pass
    
    def _send_command(self, command):
        """Send a command to the robot arm via serial."""
        if not self.connected:
            print(f"Not connected, skipping command: {command}")
            return False
        
        if self.simulation_mode:
            print(f"[SIM] Command: {command}")
            time.sleep(config.SIMULATION_DELAY)
            return True
        
        command_name = command.split()[0].upper()
        timeout = self.command_timeouts.get(command_name, 8)

        # Add newline terminator
        cmd_str = f"{command}\n"

        attempt = 0
        while attempt < self._command_retry_attempts:
            attempt += 1
            try:
                if self.verbose:
                    print(f"[SERIAL] TX (attempt {attempt}): {cmd_str.strip()}")
                self.serial_port.write(cmd_str.encode('utf-8'))
                self.serial_port.flush()

                # Wait for acknowledgment
                if self._wait_for_ack(timeout=timeout, expected_cmd=command_name):
                    if self.verbose:
                        print(f"[SERIAL] ACK received for {command_name}")
                    return True
                else:
                    print(f"No acknowledgment for command: {command} (attempt {attempt})")
                    # small pause before retry
                    time.sleep(0.2)
                    continue

            except Exception as e:
                print(f"Error sending command '{command}' on attempt {attempt}: {e}")
                time.sleep(0.2)
                continue

        return False
    
    def _wait_for_ack(self, timeout=3, expected_cmd=None):
        """Wait for acknowledgment from robot arm."""
        if self.simulation_mode:
            return True
        
        start_time = time.time()
        last_line = None
        while time.time() - start_time < timeout:
            try:
                if self.serial_port.in_waiting > 0:
                    raw = self.serial_port.readline()
                    try:
                        response = raw.decode('utf-8', errors='replace').strip()
                    except Exception:
                        response = repr(raw)
                    last_line = response
                    normalized = response.upper()

                    if self.verbose:
                        print(f"[SERIAL] RX: {response}")

                    if normalized.startswith("ERROR") or normalized.startswith("ERR"):
                        print(f"Robot error: {response}")
                        return False

                    # OK responses
                    if normalized == "OK" or normalized.startswith("OK ") or normalized.startswith("OK"):
                        # if the firmware attaches command name, optionally validate
                        if expected_cmd is None:
                            return True
                        if expected_cmd in normalized:
                            return True
                        # OK without command name — accept
                        return True

                    # ACK handling
                    if normalized.startswith("ACK"):
                        if expected_cmd is None:
                            return True
                        if f"ACK {expected_cmd}" in normalized or expected_cmd in normalized:
                            return True
                        # stale ACK — continue waiting
                        continue
                else:
                    time.sleep(0.03)
            except Exception:
                # non-fatal read/parsing error, keep waiting until timeout
                time.sleep(0.05)
                continue

        # timed out
        if self.verbose and last_line is not None:
            print(f"[SERIAL] wait_for_ack timed out, last line: {last_line}")
        return False
    
    def move_to_position(self, board_position):
        """
        Move robot arm to a specific board position (0-8).
        
        Args:
            board_position: Integer from 0 to 8 representing board position
        
        Returns:
            bool: True if successful, False otherwise
        """
        if board_position < 0 or board_position > 8:
            print(f"Invalid board position: {board_position}")
            return False
        
        print(f"Moving robot arm to position {board_position}...")
        
        if self.simulation_mode:
            # Simulate movement
            row, col = self.board_positions[board_position]
            print(f"[SIM] Moving to board position {board_position} (row={row}, col={col})")
            time.sleep(config.SIMULATION_DELAY * 2)
            print(f"[SIM] Robot moved to position {board_position}")
            return True
        
        # Send command to real robot
        command = f"MOVE {board_position}"
        success = self._send_command(command)
        
        if success:
            print(f"Robot moved to position {board_position}")
        else:
            print(f"Failed to move robot to position {board_position}")
        
        return success
    
    def pick_piece(self):
        """Pick up a game piece from storage."""
        print("Picking up game piece...")
        
        if self.simulation_mode:
            print("[SIM] Picking up piece")
            time.sleep(config.SIMULATION_DELAY)
            print("[SIM] Piece picked up")
            return True
        
        success = self._send_command("PICK")
        
        if success:
            print("Piece picked up")
        else:
            print("Failed to pick up piece")
        
        return success
    
    def place_piece(self, board_position):
        """
        Place a piece at the specified board position.
        This combines: move to position, lower gripper, release piece, return home.
        
        Args:
            board_position: Integer from 0 to 8
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"Placing piece at position {board_position}...")
        
        if self.simulation_mode:
            row, col = self.board_positions[board_position]
            print(f"[SIM] Placing piece at position {board_position} (row={row}, col={col})")
            time.sleep(config.SIMULATION_DELAY * 3)
            print(f"[SIM] Piece placed at position {board_position}")
            return True
        
        # For real robot, we can either send individual commands
        # or have the ESP32 handle the complete sequence
        command = f"PLACE {board_position}"
        success = self._send_command(command)
        
        if success:
            print(f"Piece placed at position {board_position}")
        else:
            print(f"Failed to place piece at position {board_position}")
        
        return success
    
    def home_position(self):
        """Move robot arm to home position."""
        print("Moving to home position...")
        
        if self.simulation_mode:
            print("[SIM] Moving to home position")
            time.sleep(config.SIMULATION_DELAY)
            print("[SIM] Robot at home position")
            return True
        
        success = self._send_command("HOME")
        
        if success:
            print("Robot at home position")
        else:
            print("Failed to move to home position")
        
        return success
    
    def demo_sequence(self):
        """Run a demo sequence to test robot arm."""
        print("Starting robot arm demo sequence...")
        
        if self.simulation_mode:
            print("[SIM] Running demo sequence")
            for i in range(3):
                print(f"[SIM] Demo step {i+1}")
                time.sleep(config.SIMULATION_DELAY)
            print("[SIM] Demo completed")
            return True
        
        success = self._send_command("DEMO")
        
        if success:
            print("Demo completed successfully")
        else:
            print("Demo failed")
        
        return success
    
    def close(self):
        """Close serial connection."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("Serial connection closed")
        self.connected = False
    
    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()


def test_robot_controller():
    """Test function for robot controller."""
    print("=" * 50)
    print("Testing Robot Controller")
    print("=" * 50)
    
    robot = RobotController()

    if not robot.simulation_mode and not robot.connected:
        print("\n[ERROR] Real robot mode is enabled but COM connection failed.")
        print("        Please free COM port and run again.")
        return False
    
    try:
        ok = True

        # Test home position
        ok = robot.home_position() and ok
        
        # Quick movement test (center)
        ok = robot.move_to_position(4) and ok
        time.sleep(1)
        
        # Test pick and place
        ok = robot.pick_piece() and ok
        ok = robot.place_piece(4) and ok  # Place at center
        
        # Return to home
        ok = robot.home_position() and ok
        
        if ok:
            print("\nAll tests completed successfully!")
            print("Tip: Run robot.demo_sequence() manually if you want a full stress test.")
            return True

        print("\n[ERROR] Some robot commands failed. Check firmware/protocol/COM settings.")
        return False
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    
    finally:
        robot.close()


if __name__ == "__main__":
    import sys
    ok = test_robot_controller()
    sys.exit(0 if ok else 1)