"""
ESP32 Connection Test - Test kết nối với ESP32
Chạy: python test_esp32.py
"""

import serial
import serial.tools.list_ports
import time

def list_serial_ports():
    """Liệt kê tất cả COM ports"""
    print("\n=== DANH SÁCH COM PORTS ===")
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("Không tìm thấy COM port nào!")
        print("Hãy kiểm tra:")
        print("  - ESP32 đã được kết nối qua USB")
        print("  - Driver CP210x hoặc CH340 đã được cài đặt")
        return None
    
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device}: {port.description}")
    
    return [port.device for port in ports]


def test_connection(port, baudrate=115200):
    """Test kết nối với ESP32"""
    print(f"\n=== TEST KẾT NỐI: {port} @ {baudrate} ===")
    
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        print(f"Đã mở port: {port}")
        
        # Đợi ESP32 khởi động
        time.sleep(2)
        
        # Đọc dữ liệu khởi động nếu có
        while ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"  <- {line}")
        
        # Gửi lệnh Status
        print("\nGửi lệnh 'S' (Status)...")
        ser.write(b'S\n')
        time.sleep(0.5)
        
        response = ser.readline().decode('utf-8', errors='ignore').strip()
        if response:
            print(f"  <- {response}")
            if "STATUS" in response:
                print("✓ ESP32 phản hồi thành công!")
                return ser
        else:
            print("✗ Không nhận được phản hồi")
        
        return ser
        
    except serial.SerialException as e:
        print(f"✗ Lỗi kết nối: {e}")
        return None


def test_servo_movement(ser):
    """Test di chuyển servo"""
    print("\n=== TEST DI CHUYỂN SERVO ===")
    
    commands = [
        ("H", "Home position"),
        ("M45,90,90", "Base rotate left"),
        ("M135,90,90", "Base rotate right"),
        ("M90,90,90", "Center"),
        ("G1", "Close gripper"),
        ("G0", "Open gripper"),
        ("H", "Home position"),
    ]
    
    for cmd, desc in commands:
        print(f"\n{desc}...")
        print(f"  -> {cmd}")
        ser.write(f"{cmd}\n".encode())
        time.sleep(1)
        
        # Đọc phản hồi
        while ser.in_waiting:
            response = ser.readline().decode('utf-8', errors='ignore').strip()
            if response:
                print(f"  <- {response}")
        
        input("  Nhấn Enter để tiếp tục...")


def interactive_control(ser):
    """Điều khiển tương tác"""
    print("\n=== CHẾ ĐỘ ĐIỀU KHIỂN TƯƠNG TÁC ===")
    print("Commands:")
    print("  M<a1>,<a2>,<a3> - Di chuyển đến góc")
    print("  G0              - Mở gripper")
    print("  G1              - Đóng gripper")  
    print("  H               - Về home")
    print("  S               - Status")
    print("  T               - Test sequence")
    print("  q               - Thoát")
    
    while True:
        cmd = input("\nNhập lệnh: ").strip()
        
        if cmd.lower() == 'q':
            break
        
        if cmd:
            ser.write(f"{cmd}\n".encode())
            time.sleep(0.5)
            
            while ser.in_waiting:
                response = ser.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    print(f"  <- {response}")


def main():
    print("=" * 60)
    print("     ESP32 CONNECTION TEST - TicTacToe Robot")
    print("=" * 60)
    
    # Liệt kê ports
    ports = list_serial_ports()
    
    if not ports:
        return
    
    # Chọn port
    if len(ports) == 1:
        selected_port = ports[0]
        print(f"\nTự động chọn: {selected_port}")
    else:
        idx = input("\nChọn port (số): ").strip()
        try:
            selected_port = ports[int(idx)]
        except:
            print("Lựa chọn không hợp lệ!")
            return
    
    # Test kết nối
    ser = test_connection(selected_port)
    
    if ser is None:
        print("\nKhông thể kết nối. Hãy kiểm tra:")
        print("  1. ESP32 đã được nạp firmware (esp32_firmware.ino)")
        print("  2. Baud rate đúng (115200)")
        print("  3. Port chưa bị ứng dụng khác chiếm")
        return
    
    try:
        # Menu
        while True:
            print("\n=== MENU ===")
            print("  1. Test di chuyển servo")
            print("  2. Điều khiển tương tác")
            print("  3. Test lại kết nối")
            print("  q. Thoát")
            
            choice = input("\nChọn: ").strip()
            
            if choice == '1':
                test_servo_movement(ser)
            elif choice == '2':
                interactive_control(ser)
            elif choice == '3':
                ser.write(b'S\n')
                time.sleep(0.5)
                while ser.in_waiting:
                    print(f"  <- {ser.readline().decode('utf-8', errors='ignore').strip()}")
            elif choice.lower() == 'q':
                break
    
    finally:
        ser.close()
        print("\nĐã đóng kết nối.")


if __name__ == "__main__":
    main()
