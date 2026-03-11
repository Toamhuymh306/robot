# Smart Robotic Arm Playing Tic-Tac-Toe

## Based on Unsupervised Learning Using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## 📋 Tổng quan

Dự án xây dựng hệ thống **cánh tay robot thông minh** có khả năng chơi cờ Caro (Tic-Tac-Toe) với người chơi thực. Hệ thống kết hợp:

- **Computer Vision (Thị giác máy tính)** — nhận diện bàn cờ và quân cờ từ camera
- **K-Means Clustering (Học không giám sát)** — phân loại trạng thái ô cờ
- **Minimax Algorithm với Alpha-Beta Pruning** — tính toán nước đi tối ưu
- **Cánh tay robot 3-DOF** — điều khiển bằng Inverse Kinematics

---

## 🧠 Cách Hoạt Động Chi Tiết Của Model

### Tổng quan pipeline

Hệ thống hoạt động theo một pipeline (chuỗi xử lý) tuần tự gồm **9 module**, mỗi module đảm nhận một nhiệm vụ riêng biệt:

```
Camera → Tiền xử lý ảnh → Phát hiện bàn cờ → Trích xuất đặc trưng
    → K-Means phân cụm → Ma trận trạng thái → AI Minimax
    → Ánh xạ tọa độ → Điều khiển Robot
```

```
┌──────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│  MODULE 1    │    │  MODULE 2           │    │  MODULE 3            │
│  Camera      │───▶│  Tiền Xử Lý Ảnh    │───▶│  Phát Hiện Bàn Cờ   │
│  Input       │    │  (Grayscale, Blur,  │    │  (Contour, Warp,     │
│              │    │   Canny, Morphology)│    │   Chia 9 ô)          │
└──────────────┘    └─────────────────────┘    └──────────┬───────────┘
                                                          │
                                                          ▼
┌──────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│  MODULE 9    │    │  MODULE 4           │    │  MODULE 4            │
│  Robot       │    │  Trích Xuất         │◀───│  Feature             │
│  Control     │    │  Đặc Trưng          │    │  Extraction          │
│  (IK, Servo) │    │  (HSV, RGB, Hist)   │    │  (9 ô → 9 vector)   │
└──────┬───────┘    └─────────────────────┘    └──────────┬───────────┘
       ▲                                                  │
       │                                                  ▼
┌──────┴───────┐    ┌─────────────────────┐    ┌──────────────────────┐
│  MODULE 8    │    │  MODULE 6           │    │  MODULE 5            │
│  Ánh Xạ     │    │  Ma Trận            │◀───│  K-Means             │
│  Tọa Độ     │    │  Trạng Thái         │    │  Clustering          │
│  (mm → angle)│    │  (3×3 matrix)       │    │  (3 cụm: Trống,     │
└──────────────┘    └──────────┬──────────┘    │   Người, Robot)      │
       ▲                       │               └──────────────────────┘
       │                       ▼
       │            ┌─────────────────────┐
       └────────────│  MODULE 7           │
                    │  Game AI            │
                    │  (Minimax + α-β)    │
                    └─────────────────────┘
```

---

### Module 1: Camera Input (`camera_input.py`)

**Mục đích:** Thu nhận hình ảnh từ webcam.

**Cách hoạt động:**
1. Mở webcam bằng OpenCV (`cv2.VideoCapture`)
2. Cấu hình độ phân giải (mặc định `640×480` @ `30 FPS`)
3. **Warm-up camera**: Bỏ qua 30 frame đầu tiên để camera tự động điều chỉnh ánh sáng/white balance
4. Cung cấp frame cho pipeline xử lý tiếp theo

```python
# Mỗi lần lặp, camera trả về 1 frame
ret, frame = camera.get_frame()  # frame là ảnh BGR (640×480×3)
```

---

### Module 2: Image Preprocessing (`image_preprocessing.py`)

**Mục đích:** Tiền xử lý ảnh để chuẩn bị cho bước phát hiện bàn cờ.

**Chuỗi xử lý:**

```
Ảnh gốc (BGR) → Grayscale → Gaussian Blur → Canny Edge → Morphology → Contours
```

| Bước | Kỹ thuật | Mô tả |
|------|----------|-------|
| 1. Grayscale | `cv2.cvtColor(BGR2GRAY)` | Chuyển ảnh màu sang ảnh xám (1 kênh) |
| 2. Gaussian Blur | Kernel `5×5` | Làm mờ để giảm nhiễu, tránh phát hiện cạnh giả |
| 3. Canny Edge | Ngưỡng `50–150` | Phát hiện cạnh dựa trên gradient — tìm đường viền bàn cờ |
| 4. Dilation | 2 lần lặp | Làm dày các cạnh, nối các cạnh bị đứt gãy |
| 5. Erosion | 1 lần lặp | Loại bỏ nhiễu nhỏ sau khi dilate |
| 6. Find Contours | `RETR_TREE` | Tìm tất cả các đường viền (contour) trong ảnh |

**Ngoài ra**, module còn hỗ trợ:
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Tăng cường tương phản khi ánh sáng yếu
- **Adaptive Threshold**: Ngưỡng thích ứng cho trường hợp ánh sáng không đều
- **Perspective Transform**: Chuyển đổi góc nhìn (biến hình thang thành hình vuông)

---

### Module 3: Board Detection (`board_detection.py`)

**Mục đích:** Phát hiện bàn cờ 3×3 trong ảnh camera và cắt ra 9 ô riêng biệt.

**Cách hoạt động chi tiết:**

#### Bước 1: Tìm bàn cờ
```
Contours → Lọc theo diện tích → Xấp xỉ đa giác → Chỉ lấy tứ giác → Kiểm tra tỷ lệ
```
- Quét tất cả contour, lọc theo **diện tích** (`10,000 – 200,000 pixel²`)
- Xấp xỉ contour thành đa giác (`cv2.approxPolyDP`), chỉ giữ **tứ giác** (4 cạnh)
- Kiểm tra **tỷ lệ khung hình** (aspect ratio `0.8–1.2`) để đảm bảo là hình vuông
- Chọn tứ giác có **diện tích lớn nhất** → đó là bàn cờ

#### Bước 2: Perspective Warp (Chuyển đổi góc nhìn)
```
4 góc bàn cờ → Sắp xếp → Tính ma trận biến đổi M → Warp thành hình vuông 300×300
```
- Camera có thể nhìn bàn cờ từ một góc nghiêng → bàn cờ sẽ bị **biến dạng** (hình thang)
- Dùng **Perspective Transform** (`cv2.getPerspectiveTransform`) để "duỗi thẳng" bàn cờ thành hình vuông hoàn hảo `300×300 pixel`
- 4 góc được sắp xếp theo thứ tự: trên-trái → trên-phải → dưới-phải → dưới-trái

#### Bước 3: Chia thành 9 ô
```
Ảnh 300×300 → Chia 3×3 → 9 ô, mỗi ô 100×100 pixel (trừ padding 10px)
```
- Mỗi ô có kích thước `(100 - 2×10) = 80×80 pixel` (sau khi bỏ padding)
- Padding giúp tránh lấy phần đường kẻ vào ô

---

### Module 4: Feature Extraction (`feature_extraction.py`)

**Mục đích:** Trích xuất **vector đặc trưng màu sắc** từ mỗi ô cờ để K-Means có thể phân loại.

**Tại sao cần trích xuất đặc trưng?**
- K-Means không hiểu ảnh trực tiếp → cần chuyển mỗi ô thành **một vector số** đại diện cho màu sắc
- Ô trống, quân người chơi, quân robot sẽ có màu **khác nhau** → vector đặc trưng cũng khác nhau

**Các loại đặc trưng hỗ trợ:**

| Loại đặc trưng | Kích thước vector | Mô tả |
|-----------------|-------------------|-------|
| `hsv_mean` (mặc định) | 3 | Trung bình H, S, V của mỗi ô |
| `rgb_mean` | 3 | Trung bình B, G, R |
| `histogram` | 24 | Histogram 8 bins × 3 kênh BGR |
| `hsv_histogram` | 24 | Histogram 8 bins × 3 kênh HSV |
| `combined` | 6 | RGB mean + HSV mean |
| `color_stats` | 9 | RGB mean + RGB std + HSV mean |

**Mặc định dùng `hsv_mean`** vì:
- **H (Hue)**: Phân biệt màu sắc (đỏ, xanh, ...) — quan trọng nhất
- **S (Saturation)**: Độ bão hòa — ô trống thường có S thấp (xám), quân cờ có S cao (màu đậm)
- **V (Value)**: Độ sáng

```python
# Ví dụ đặc trưng HSV mean cho 3 loại ô:
Ô trống:     [90, 30, 150]   # Hue trung tính, Saturation thấp
Quân xanh:   [60, 200, 100]  # Hue = 60 (xanh lá), Saturation cao
Quân đỏ:     [5, 220, 120]   # Hue = 5 (đỏ), Saturation cao
```

**Kết quả:** 9 ô → ma trận đặc trưng `(9, 3)` — mỗi hàng là vector đặc trưng của 1 ô.

---

### Module 5: K-Means Clustering (`kmeans_classifier.py`) ⭐ *LÕI CỦA MODEL*

**Mục đích:** Phân loại 9 ô cờ thành 3 nhóm (Trống / Quân người / Quân robot) bằng **học không giám sát (Unsupervised Learning)**.

**Tại sao dùng K-Means (Unsupervised) thay vì Supervised Learning?**
- **Không cần dữ liệu huấn luyện gán nhãn** — hệ thống tự phân nhóm
- **Thích ứng** với nhiều loại quân cờ khác nhau (nắp chai, giấy màu, ...)
- **Linh hoạt** với điều kiện ánh sáng thay đổi

#### Thuật toán K-Means hoạt động như thế nào?

```
Dữ liệu: 9 vector đặc trưng (mỗi vector 3 chiều: H, S, V)
Mục tiêu: Chia 9 vector thành K=3 nhóm (cluster) sao cho các vector trong cùng nhóm gần nhau nhất
```

**Các bước của K-Means:**

```
1. Chọn ngẫu nhiên 3 điểm làm centroid ban đầu
2. Lặp lại đến khi hội tụ (tối đa 100 lần):
   a. GÁN NHÃN: mỗi điểm được gán vào cluster có centroid gần nhất
   b. CẬP NHẬT: tính lại centroid = trung bình tất cả điểm trong cluster
3. Trả về nhãn cluster cho mỗi điểm
```

Minh họa bằng số:
```
Input: 9 vector HSV mean
  Ô 0: [90, 30, 150]  ──┐
  Ô 1: [85, 25, 145]  ──┤── Cluster 0 (màu xám, saturation thấp) → TRỐNG
  Ô 2: [95, 35, 155]  ──┤
  Ô 3: [88, 28, 148]  ──┤
  Ô 4: [92, 32, 152]  ──┘
  Ô 5: [60, 200, 100] ──┐── Cluster 1 (hue cao, saturation cao) → NGƯỜI CHƠI
  Ô 6: [55, 190, 95]  ──┘
  Ô 7: [5, 220, 120]  ──┐── Cluster 2 (hue thấp, saturation cao) → ROBOT
  Ô 8: [175, 210, 115]──┘
```

#### Cách ánh xạ Cluster → Nhãn thực tế

Sau khi K-Means chia thành 3 cluster, hệ thống cần xác định cluster nào tương ứng với loại ô nào:

1. **Cluster TRỐNG**: Cluster có **số lượng phần tử nhiều nhất** (đầu game, 5-7 ô trống)
2. **Cluster NGƯỜI CHƠI vs ROBOT**: Phân biệt dựa trên giá trị **Hue trung bình**:
   - Hue thấp (gần 0 hoặc gần 170) → **đỏ** → Robot
   - Hue cao (gần 60) → **xanh** → Người chơi

#### Chuẩn hóa dữ liệu (StandardScaler)
- Trước khi đưa vào K-Means, dữ liệu được **chuẩn hóa** bằng `StandardScaler` (trung bình = 0, độ lệch chuẩn = 1)
- Đảm bảo tất cả các chiều đặc trưng có **trọng số bằng nhau** trong tính khoảng cách

#### Adaptive K-Means (Mở rộng)
- Hệ thống có lớp `AdaptiveKMeansClassifier` để **cập nhật dần** khi thu thập thêm dữ liệu
- Dùng `learning_rate = 0.1` và lưu tối đa 100 sample trong lịch sử
- Giúp thích ứng khi ánh sáng thay đổi trong quá trình chơi

#### Tham số cấu hình K-Means

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `N_CLUSTERS` | 3 | 3 nhóm: Trống, Người chơi, Robot |
| `KMEANS_MAX_ITERATIONS` | 100 | Tối đa 100 vòng lặp hội tụ |
| `KMEANS_TOLERANCE` | 1e-4 | Ngưỡng dừng khi centroid thay đổi < 0.0001 |
| `KMEANS_RANDOM_STATE` | 42 | Seed cố định để kết quả lặp lại được |
| `n_init` | 10 | Chạy 10 lần với centroid khởi tạo khác nhau, lấy kết quả tốt nhất |

---

### Module 6: Board State (`board_state.py`)

**Mục đích:** Quản lý trạng thái bàn cờ dưới dạng **ma trận 3×3**.

**Biểu diễn:**
```
Ma trận 3×3 với giá trị:
   0 = Ô trống
   1 = Quân Robot (O)
  -1 = Quân Người chơi (X)

Ví dụ:
┌───┬───┬───┐
│ O │   │ X │    Ma trận: [[ 1,  0, -1],
├───┼───┼───┤              [ 0,  1,  0],
│   │ O │   │              [-1,  0,  1]]
├───┼───┼───┤
│ X │   │ O │
└───┴───┴───┘
```

**Chức năng chính:**
- **`update_from_labels(labels)`**: Nhận mảng 9 nhãn từ K-Means → reshape thành 3×3
- **`detect_new_move(new_labels)`**: So sánh với trạng thái trước → phát hiện nước đi mới
- **`check_winner()`**: Kiểm tra 8 đường thẳng (3 hàng + 3 cột + 2 đường chéo), nếu tổng = ±3 → có người thắng
- **`is_game_over()`**: Kết thúc nếu có người thắng hoặc hết ô trống (hòa)
- **Lưu lịch sử**: Mỗi nước đi được lưu trong `history` để có thể hoàn tác

---

### Module 7: Game AI — Minimax (`game_ai.py`)

**Mục đích:** Tìm **nước đi tối ưu** cho robot bằng thuật toán **Minimax với Alpha-Beta Pruning**.

#### Thuật toán Minimax là gì?

Minimax mô phỏng **tất cả các khả năng** xảy ra trong game và chọn nước đi tốt nhất:

```
Robot muốn TỐI ĐA hóa điểm (Maximizer)
Người chơi muốn TỐI THIỂU hóa điểm (Minimizer)

Cây game:
                    Robot đi
                   /    |    \
              Người     Người    Người
              /  \      /  \     /  \
           Robot Robot Robot Robot Robot
           ...  ...   ...  ...   ...
```

**Quy tắc tính điểm:**
| Trạng thái | Điểm |
|------------|-------|
| Robot thắng | `+100 + depth` (thắng sớm → điểm cao hơn) |
| Người chơi thắng | `-100 - depth` (thua muộn → ít bất lợi hơn) |
| Hòa | `0` |
| Robot có 2 quân + 1 ô trống | `+10` |
| Người chơi có 2 quân + 1 ô trống | `-10` (cần chặn!) |
| Chiếm ô trung tâm | `+3` |
| Chiếm ô góc | `+2` |

#### Alpha-Beta Pruning (Tỉa cành)

Không cần duyệt **tất cả** nhánh cây:
- **Alpha**: Giá trị tốt nhất mà Maximizer (Robot) ĐÃ tìm được
- **Beta**: Giá trị tốt nhất mà Minimizer (Người chơi) ĐÃ tìm được
- Nếu `beta ≤ alpha` → **cắt bỏ** nhánh hiện tại (không cần duyệt thêm)
- Giảm số node đánh giá từ **9! = 362,880** xuống chỉ vài trăm

```python
# Ví dụ: Nếu Robot đã tìm được nước đi cho 100 điểm (alpha=100),
# mà nhánh hiện tại Người chơi có thể ép xuống -100 (beta=-100),
# thì beta ≤ alpha → cắt bỏ, không cần xét thêm.
```

#### Chiến lược đặc biệt
- **Bàn cờ trống** → luôn đi **ô trung tâm** (1,1) — vị trí tốt nhất
- **Quick check** (`get_immediate_win_or_block`): Trước khi chạy Minimax, kiểm tra nhanh:
  1. Có thể thắng ngay? → Thắng!
  2. Đối thủ sắp thắng? → Chặn!

---

### Module 8: Coordinate Mapping (`coordinate_mapping.py`)

**Mục đích:** Chuyển đổi vị trí ô cờ `(hàng, cột)` → tọa độ vật lý `(x, y, z)` mm cho robot.

**Cách tính:**
```python
# Gốc bàn cờ: (100, 0, 0) mm so với chân robot
# Khoảng cách giữa ô: 50mm
x = origin_x + (col - 1) × 50   # col: 0,1,2
y = origin_y + (1 - row) × 50   # row: 0,1,2 (đảo ngược trục Y)
z = origin_z                     # Mặt phẳng bàn cờ
```

**Bảng tọa độ 9 ô (mm):**
```
Ô (0,0): (50, 50, 0)   |  Ô (0,1): (100, 50, 0)  |  Ô (0,2): (150, 50, 0)
Ô (1,0): (50, 0, 0)    |  Ô (1,1): (100, 0, 0)    |  Ô (1,2): (150, 0, 0)
Ô (2,0): (50, -50, 0)  |  Ô (2,1): (100, -50, 0)  |  Ô (2,2): (150, -50, 0)
```

**Quỹ đạo Pick-and-Place** (trajectory gồm 8 bước):
1. Di chuyển đến trên kho quân cờ (z + 40mm)
2. Hạ xuống vị trí lấy quân
3. Đóng kẹp (gripper)
4. Nâng lên (z + 60mm)
5. Di chuyển đến trên ô mục tiêu (z + 40mm)
6. Hạ xuống ô mục tiêu
7. Mở kẹp (thả quân)
8. Nâng lên (an toàn)

---

### Module 9: Robot Control (`robot_control.py`)

**Mục đích:** Điều khiển cánh tay robot 3 bậc tự do (3-DOF) bằng **Inverse Kinematics**.

#### Cấu hình cánh tay robot

```
         [End Effector / Gripper]
              ╱
    L3=100mm ╱  θ3 (Elbow)
            ╱
           ●─────────
           │ L2=100mm   θ2 (Shoulder)
           │
           │ L1=50mm
           │
    ═══════●═══════    θ1 (Base rotation)
        Robot Base
```

#### Inverse Kinematics (Động học ngược)

**Bài toán:** Cho vị trí mong muốn `(x, y, z)` → tính góc servo `(θ1, θ2, θ3)`.

```python
# Bước 1: Góc xoay đế (θ1)
θ1 = atan2(y, x)

# Bước 2: Chiếu lên mặt phẳng 2D (r-z)
r = √(x² + y²)           # Khoảng cách ngang
z_adj = z - L1            # Trừ độ cao đế

# Bước 3: Khoảng cách từ vai đến mục tiêu
d = √(r² + z_adj²)

# Bước 4: Góc khuỷu tay (θ3) — Định lý Cosine
cos(θ3) = (d² - L2² - L3²) / (2 × L2 × L3)
θ3 = acos(cos(θ3))

# Bước 5: Góc vai (θ2)
α = atan2(z_adj, r)
β = acos((L2² + d² - L3²) / (2 × L2 × d))
θ2 = α + β

# Bước 6: Chuyển sang góc servo (0°–180°)
servo1 = 90 + θ1    # Giữa tại 90°
servo2 = 90 + θ2
servo3 = 180 - θ3   # Đảo ngược cho khuỷu tay
```

**Kiểm tra tầm với:**
- `d > L2 + L3` → quá xa, không với tới
- `d < |L2 - L3|` → quá gần

#### Giao tiếp Serial
- Gửi lệnh qua UART đến **ESP32/Arduino**: `M{θ1},{θ2},{θ3}\n`
- Lệnh gripper: `G1\n` (đóng), `G0\n` (mở)
- Baudrate: `115200` (ESP32) hoặc `9600` (Arduino)
- Hỗ trợ **Simulation Mode** — chạy không cần phần cứng thật

#### Di chuyển mượt (Smooth Movement)
- Nội suy tuyến tính giữa vị trí hiện tại và vị trí đích
- Chia thành nhiều bước nhỏ (tùy khoảng cách) → di chuyển mượt mà

---

## 🔄 Luồng Hoạt Động Của Một Ván Chơi

```
┌─────────────────────────────────────────────────────────┐
│  1. KHỞI TẠO                                           │
│     ├── Mở camera → warmup                              │
│     ├── Kết nối robot (hoặc chế độ simulation)           │
│     └── Robot về vị trí Home (90°, 90°, 90°)            │
├─────────────────────────────────────────────────────────┤
│  2. CALIBRATE BÀN CỜ                                   │
│     ├── Camera tìm bàn cờ (tối đa 50 frame)             │
│     └── Người dùng nhấn 'c' để xác nhận                 │
├─────────────────────────────────────────────────────────┤
│  3. VÒNG LẶP CHÍNH                                     │
│     ├── LƯỢT NGƯỜI CHƠI:                                │
│     │   ├── Người đặt quân lên bàn cờ                    │
│     │   ├── Nhấn 'n' để xác nhận                         │
│     │   ├── Camera chụp → Pipeline xử lý                 │
│     │   │   ├── Tiền xử lý ảnh                           │
│     │   │   ├── Phát hiện bàn cờ → Warp → 9 ô           │
│     │   │   ├── Trích xuất HSV mean (9 vector)           │
│     │   │   ├── K-Means phân cụm (3 nhóm)               │
│     │   │   └── Cập nhật ma trận trạng thái              │
│     │   └── Kiểm tra thắng/thua/hòa                     │
│     │                                                    │
│     └── LƯỢT ROBOT:                                      │
│         ├── Minimax tính nước đi tối ưu                  │
│         ├── Ánh xạ (hàng,cột) → tọa độ (x,y,z) mm      │
│         ├── Inverse Kinematics → góc servo (θ1,θ2,θ3)   │
│         ├── Robot gắp quân từ kho → đặt lên ô           │
│         └── Kiểm tra thắng/thua/hòa                     │
├─────────────────────────────────────────────────────────┤
│  4. KẾT THÚC                                           │
│     ├── Hiển thị kết quả (Robot/Người thắng/Hòa)        │
│     ├── Hỏi chơi lại → vòng lặp mới                     │
│     └── Thoát → shutdown hệ thống                        │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 K-Means Classification — Tóm tắt

| Cluster | Ý nghĩa | Đặc điểm nhận dạng |
|---------|---------|---------------------|
| 0 | Ô trống | Saturation thấp (màu xám), số lượng nhiều nhất |
| 1 | Quân Robot | Hue thấp (đỏ), Saturation cao |
| -1 | Quân Người chơi | Hue cao (xanh), Saturation cao |

**Đặc trưng mặc định:** HSV mean (3 chiều: Hue, Saturation, Value)

---

## 📁 Cấu Trúc Dự Án

```
TicTacToe-Robot-KMeans/
├── modules/
│   ├── camera_input.py        # Module 1: Thu nhận ảnh dari camera
│   ├── image_preprocessing.py # Module 2: Tiền xử lý ảnh
│   ├── board_detection.py     # Module 3: Phát hiện bàn cờ
│   ├── feature_extraction.py  # Module 4: Trích xuất đặc trưng màu
│   ├── kmeans_classifier.py   # Module 5: Phân loại K-Means ⭐
│   ├── board_state.py         # Module 6: Quản lý trạng thái bàn cờ
│   ├── game_ai.py             # Module 7: AI Minimax + Alpha-Beta
│   ├── coordinate_mapping.py  # Module 8: Ánh xạ tọa độ
│   └── robot_control.py       # Module 9: Điều khiển robot
├── docs/
│   └── SYSTEM_ARCHITECTURE.md # Tài liệu kiến trúc hệ thống
├── test_data/                 # Dữ liệu test đã thu thập
├── config.py                  # Cấu hình tham số hệ thống
├── main.py                    # Script tích hợp chính
├── demo.py                    # Script demo và kiểm thử
├── test_camera.py             # Test camera
├── test_paper_board.py        # Test nhận diện bàn cờ giấy (với trackbars)
├── test_accuracy.py           # Test độ chính xác K-means
├── test_esp32.py              # Test kết nối ESP32
├── requirements.txt           # Thư viện Python cần thiết
├── arduino_firmware.ino       # Firmware Arduino UNO/Nano
├── esp32_firmware.ino         # Firmware ESP32
└── README.md                  # File này
```

---

## 🚀 Hướng Dẫn Chạy

### Yêu cầu
- Python 3.8 trở lên
- Webcam (cho nhận diện realtime)
- ESP32/Arduino + cánh tay robot 3-DOF (tùy chọn, có chế độ simulation)

### Cài đặt

```bash
cd d:/robot
pip install -r requirements.txt
```

### Test camera
```bash
python test_camera.py
```

### Chạy demo
```bash
python demo.py          # Tất cả demo
python demo.py board    # Demo phát hiện bàn cờ
python demo.py kmeans   # Demo phân loại K-Means
python demo.py ai       # Demo Game AI
python demo.py robot    # Demo mô phỏng robot
python demo.py game     # Demo game đầy đủ
python demo.py camera   # Demo camera trực tiếp
```

### Chạy hệ thống đầy đủ
```bash
python main.py
```

### Test độ chính xác K-Means
```bash
python test_accuracy.py
# Chọn 1: Thu thập data (chụp ảnh + gán nhãn)
# Chọn 2: Test accuracy trên data đã thu thập
# Chọn 3: Test realtime
```

### Test kết nối ESP32
```bash
python test_esp32.py
```

### Phím điều khiển
- **`n`** — Xác nhận nước đi
- **`r`** — Reset game
- **`q`** — Thoát

---

## 🔧 Cấu hình (`config.py`)

```python
# Camera
CAMERA_ID = 0              # Index camera (0 = webcam mặc định)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# K-Means
N_CLUSTERS = 3             # 3 nhóm: Trống, Người chơi, Robot
FEATURE_TYPE = 'hsv_mean'  # Loại đặc trưng

# Robot
SIMULATION_MODE = True     # True = chạy mô phỏng, False = dùng robot thật
SERIAL_PORT = 'COM3'       # Cổng COM của ESP32
SERIAL_BAUDRATE = 115200   # Baudrate: 115200 (ESP32), 9600 (Arduino)

# Kích thước robot arm (mm)
LINK_1_LENGTH = 50         # Đế đến vai
LINK_2_LENGTH = 100        # Vai đến khuỷu tay
LINK_3_LENGTH = 100        # Khuỷu tay đến gripper
```

---

## 🔌 Cài Đặt Phần Cứng

### Linh kiện cần thiết
1. **ESP32** (khuyên dùng) hoặc Arduino UNO/Nano
2. 3× Servo motor (SG90 hoặc MG996R)
3. Webcam hoặc camera laptop
4. Nguồn 5V 2A riêng cho servo
5. Cánh tay robot 3-DOF
6. Bàn cờ giấy 15-20cm

### Sơ đồ nối dây ESP32
```
ESP32          Servo
-----          -----
GPIO 13  →     Servo 1 (Base)
GPIO 12  →     Servo 2 (Shoulder)
GPIO 14  →     Servo 3 (Elbow)
GPIO 27  →     Servo 4 (Gripper)
GND      →     GND chung

⚠️ Nguồn 5V riêng cho servo, KHÔNG lấy từ ESP32!
```

### Sơ đồ nối dây Arduino (thay thế)
```
Arduino:
  - Pin 9  → Servo 1 (Base)
  - Pin 10 → Servo 2 (Shoulder)
  - Pin 11 → Servo 3 (Elbow)
  - Pin 6  → Servo 4 (Gripper)
```

---

## 📝 Làm Bàn Cờ Test

1. Cắt giấy trắng hình vuông (15-20cm)
2. Kẻ lưới 3×3 bằng bút đen (đậm ~2mm)
3. Dùng vật thể màu làm quân cờ:
   - **X**: Nắp chai đỏ, giấy đỏ
   - **O**: Nắp chai xanh, giấy xanh

**Mẹo nhận diện tốt hơn:**
- Nền tối (bàn gỗ, vải đen)
- Ánh sáng đều, không bóng đổ
- Camera đặt thẳng hoặc nghiêng nhẹ

---

## 📊 Bảng Đánh Giá Độ Chính Xác

| Accuracy | Trạng thái |
|----------|-----------|
| < 70% | Cần cải thiện quân cờ/ánh sáng |
| 70–85% | Chấp nhận được |
| 85–95% | Tốt |
| > 95% | Xuất sắc |

---

## 📱 Triển Khai

### Raspberry Pi
```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-numpy
pip3 install scikit-learn pyserial
```

### Jetson Nano
```bash
pip3 install opencv-python numpy scikit-learn pyserial
```

---

## 🔮 Hướng Phát Triển

1. **Deep Learning**: Thay K-Means bằng CNN để nhận diện quân cờ chính xác hơn
2. **Reinforcement Learning**: Huấn luyện AI qua tự chơi (self-play)
3. **Voice Control**: Thêm điều khiển bằng giọng nói
4. **6-DOF Support**: Hỗ trợ cánh tay robot phức tạp hơn
5. **Online Learning**: Tự thích ứng khi ánh sáng thay đổi

---

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests.

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

_Built with ❤️ for AI and Robotics Research_
