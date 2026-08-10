"""
============================================================
  face_recognition_pipeline.py
  Pipeline nhận diện khuôn mặt theo thời gian thực
  Sử dụng: InsightFace + ONNX Runtime (CPU)
============================================================

Luồng hoạt động:
  1. [STARTUP]  load_database()   → Quét thư mục, trích xuất embeddings,
                                    lưu vào RAM dưới dạng NumPy arrays.
  2. [STARTUP]  init_model()      → Khởi tạo InsightFace với ONNX Runtime
                                    được tối ưu đa luồng.
  3. [RUNTIME]  main_loop()       → Đọc từng frame webcam, gọi process_frame(),
                                    hiển thị kết quả.
  4. [RUNTIME]  process_frame()   → Detect → Embed → Cosine Similarity → Label.

Cấu trúc thư mục:
  smartCam/
  ├── face_recognition_pipeline.py   ← file này
  ├── requirements.txt
  └── face_recognition/
      └── database/
          ├── nguyen_van_a/
          │   ├── photo1.jpg
          │   └── photo2.jpg
          └── tran_thi_b/
              └── photo1.png
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

# ============================================================
# SECTION 1: CẤU HÌNH TOÀN CỤC (Global Configuration)
# ============================================================

# --- Đường dẫn ---
DATABASE_DIR = Path(__file__).parent / "face_recognition" / "database"

# --- Model InsightFace ---
# "buffalo_sc": nhẹ, nhanh, phù hợp CPU thông thường
# "buffalo_l" : nặng hơn, chính xác hơn, cần CPU mạnh hơn
MODEL_NAME = "buffalo_sc"

# --- Tối ưu ONNX Runtime ---
# Số thread vật lý/logic dành cho ONNX inference
INTRA_OP_NUM_THREADS = 8

# --- Nhận diện ---
# Ngưỡng Cosine Similarity [0.0 - 1.0]
# > threshold → cùng người | ≤ threshold → Unknown
# buffalo_sc: 0.4 khuyến nghị; buffalo_l: 0.5
COSINE_THRESHOLD = 0.4

# --- Camera ---
CAMERA_INDEX = 0          # 0 = webcam mặc định
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# --- Hiển thị ---
# Kích thước ảnh input cho model detection (0 = dùng mặc định insightface)
DET_SIZE = (640, 640)

# Định dạng ảnh được hỗ trợ trong database
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ============================================================
# SECTION 2: KHỞI TẠO MODEL (Model Initialization)
# ============================================================

def build_onnx_session_options() -> ort.SessionOptions:
    """
    Tạo ONNX Runtime SessionOptions được tối ưu cho CPU.

    Các tối ưu áp dụng:
      - intra_op_num_threads: Số luồng CPU tối đa cho phép
        mỗi operator ONNX sử dụng (tận dụng đa nhân).
      - graph_optimization_level = ORT_ENABLE_ALL: Bật TẤT CẢ
        các tối ưu đồ thị (operator fusion, constant folding,
        memory layout tối ưu, v.v.).

    Returns:
        ort.SessionOptions đã được cấu hình.
    """
    opts = ort.SessionOptions()

    # Số thread nội bộ cho mỗi op (parallelism trong 1 operator)
    opts.intra_op_num_threads = INTRA_OP_NUM_THREADS

    # Bật toàn bộ tối ưu đồ thị: quan trọng nhất để giảm latency
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Tắt log verbose của ONNX Runtime (chỉ giữ ERROR)
    opts.log_severity_level = 3  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR

    log.info(
        f"ONNX SessionOptions: intra_op_threads={INTRA_OP_NUM_THREADS}, "
        f"graph_opt=ORT_ENABLE_ALL"
    )
    return opts


def init_model() -> FaceAnalysis:
    """
    Khởi tạo InsightFace FaceAnalysis với ONNX Runtime tối ưu.

    InsightFace sử dụng ONNX Runtime nội bộ để chạy các model
    detection và recognition. Ta inject SessionOptions tối ưu
    vào thông qua providers_options.

    Returns:
        FaceAnalysis đã sẵn sàng (đã gọi prepare()).
    """
    log.info(f"Đang khởi tạo model InsightFace: {MODEL_NAME} ...")

    sess_opts = build_onnx_session_options()

    # providers: Chỉ dùng CPUExecutionProvider (không cần GPU)
    # providers_options: Truyền SessionOptions vào mỗi provider
    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"],
        # Lưu ý: InsightFace nhận providers_options dạng list of dict,
        # session_options được truyền riêng khi build session nội bộ.
        # Ta set global session options qua ort.set_default_logger_severity
        # và sử dụng allowed_modules để kiểm soát các sub-model.
        allowed_modules=["detection", "recognition", "landmark_3d_68"],
    )

    # Đặt kích thước ảnh input cho detector
    # det_size=(640,640): cân bằng tốc độ và độ chính xác
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    # ---- Inject SessionOptions vào các session ONNX nội bộ ----
    # InsightFace tạo session ONNX cho từng model con (detector, recognizer).
    # Ta cập nhật intra_op_num_threads trực tiếp trên các session đã tạo.
    _patch_insightface_sessions(app, sess_opts)

    log.info("Model sẵn sàng!")
    return app


def _patch_insightface_sessions(app: FaceAnalysis, opts: ort.SessionOptions):
    """
    Hàm nội bộ: Cập nhật cấu hình ONNX Runtime cho các session
    đã được InsightFace khởi tạo nội bộ.

    InsightFace không expose SessionOptions trực tiếp, nên ta
    duyệt qua các model con và set lại intra_op_num_threads.

    Args:
        app:  FaceAnalysis instance đã gọi prepare().
        opts: SessionOptions với cấu hình tối ưu.
    """
    patched_count = 0
    for model in app.models.values():
        # Mỗi model InsightFace có attribute 'session' là ort.InferenceSession
        session = getattr(model, "session", None)
        if session is not None and hasattr(session, "_sess"):
            try:
                # Cập nhật intra_op_num_threads thông qua C binding
                session._sess.set_intra_op_num_threads(opts.intra_op_num_threads)
                patched_count += 1
            except Exception:
                # Không phải tất cả phiên bản ONNX Runtime hỗ trợ; bỏ qua
                pass
    if patched_count > 0:
        log.info(f"Đã patch {patched_count} ONNX session(s) với {INTRA_OP_NUM_THREADS} threads.")
    else:
        log.warning(
            "Không thể patch ONNX sessions nội bộ (phiên bản InsightFace không hỗ trợ). "
            "Cấu hình threads vẫn có hiệu lực qua environment variables."
        )


# ============================================================
# SECTION 3: TẢI CƠ SỞ DỮ LIỆU (Database Loading)
# ============================================================

class FaceDatabase:
    """
    Cơ sở dữ liệu khuôn mặt được load HOÀN TOÀN vào RAM khi khởi động.

    Attributes:
        embeddings (np.ndarray): Ma trận (N, 512) chứa tất cả embedding vectors.
                                  Được normalize L2 để tính Cosine Similarity
                                  nhanh bằng dot product.
        labels     (list[str]): Danh sách N nhãn (tên người) tương ứng.
        is_empty   (bool):      True nếu database không có dữ liệu hợp lệ.
    """

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None  # (N, 512) float32
        self.labels: list[str] = []
        self.is_empty: bool = True


def load_database(app: FaceAnalysis) -> FaceDatabase:
    """
    Hàm chạy ĐỘC LẬP lúc khởi động: Quét DATABASE_DIR, trích xuất
    embeddings của từng người và lưu vào RAM.

    Quy trình:
      1. Duyệt từng thư mục con trong DATABASE_DIR (mỗi thư mục = 1 người).
      2. Với mỗi ảnh, trích xuất embedding 512-chiều bằng InsightFace.
      3. Nếu có nhiều ảnh → tính embedding TRUNG BÌNH → robust hơn.
      4. Normalize L2 tất cả vectors → dot product = cosine similarity.
      5. Stack thành numpy array (N, 512) lưu vào RAM.

    Args:
        app: FaceAnalysis instance đã được khởi tạo.

    Returns:
        FaceDatabase với embeddings và labels đã nạp.
    """
    db = FaceDatabase()

    if not DATABASE_DIR.exists():
        log.error(f"Thư mục database không tồn tại: {DATABASE_DIR}")
        return db

    person_dirs = [d for d in DATABASE_DIR.iterdir() if d.is_dir()]

    if not person_dirs:
        log.warning(f"Không tìm thấy thư mục con nào trong {DATABASE_DIR}")
        return db

    log.info(f"Bắt đầu load database: {len(person_dirs)} người từ {DATABASE_DIR}")

    all_embeddings = []
    all_labels     = []

    for person_dir in tqdm(person_dirs, desc="Loading database", unit="person"):
        person_name = person_dir.name

        # Lấy danh sách ảnh hợp lệ trong thư mục của người này
        image_files = [
            f for f in person_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not image_files:
            log.warning(f"  [{person_name}] Không có ảnh nào → bỏ qua.")
            continue

        person_embeddings = []

        for img_path in image_files:
            embedding = _extract_embedding_from_file(app, img_path)
            if embedding is not None:
                person_embeddings.append(embedding)

        if not person_embeddings:
            log.warning(f"  [{person_name}] Không trích xuất được embedding nào → bỏ qua.")
            continue

        # Tính embedding trung bình nếu có nhiều ảnh (tăng độ ổn định)
        avg_embedding = np.mean(person_embeddings, axis=0)  # (512,)

        # Normalize L2: ||v|| = 1  →  dot(v1, v2) = cosine_similarity(v1, v2)
        norm = np.linalg.norm(avg_embedding)
        if norm > 1e-6:
            avg_embedding = avg_embedding / norm

        all_embeddings.append(avg_embedding)
        all_labels.append(person_name)

        log.info(f"  [{person_name}] ✓ {len(person_embeddings)}/{len(image_files)} ảnh → 1 embedding.")

    if not all_embeddings:
        log.error("Database rỗng! Không thể nhận diện bất kỳ ai.")
        return db

    # Stack thành ma trận (N, 512) để tính batch cosine similarity nhanh
    db.embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)
    db.labels     = all_labels
    db.is_empty   = False

    log.info(
        f"Database đã tải xong: {len(db.labels)} người, "
        f"ma trận embedding shape={db.embeddings.shape}"
    )
    return db


def _extract_embedding_from_file(
    app: FaceAnalysis,
    img_path: Path
) -> Optional[np.ndarray]:
    """
    Hàm nội bộ: Đọc ảnh từ file, detect khuôn mặt,
    trả về embedding vector (512,) của khuôn mặt lớn nhất.

    Chỉ dùng lúc khởi động (load_database), KHÔNG gọi trong vòng lặp camera.

    Args:
        app:      FaceAnalysis instance.
        img_path: Đường dẫn file ảnh.

    Returns:
        np.ndarray (512,) hoặc None nếu không detect được khuôn mặt.
    """
    try:
        # Đọc ảnh bằng OpenCV (BGR)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            log.warning(f"    Không đọc được: {img_path.name}")
            return None

        # InsightFace nhận BGR numpy array
        faces = app.get(img_bgr)

        if not faces:
            log.warning(f"    Không phát hiện khuôn mặt: {img_path.name}")
            return None

        # Chọn khuôn mặt có bounding box lớn nhất (thường là khuôn mặt chính)
        main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # embedding: vector 512 chiều (đã normalize bởi InsightFace recognition model)
        return main_face.embedding.copy()  # (512,) float32

    except Exception as e:
        log.error(f"    Lỗi khi xử lý {img_path.name}: {e}")
        return None


# ============================================================
# SECTION 4: XỬ LÝ FRAME THỜI GIAN THỰC (Real-time Inference)
# ============================================================

def cosine_similarity_batch(query_emb: np.ndarray, db_embs: np.ndarray) -> np.ndarray:
    """
    Tính Cosine Similarity giữa 1 query vector và toàn bộ database
    bằng phép nhân ma trận (nhanh hơn vòng lặp Python ~100x).

    Vì tất cả vectors đã được normalize L2 (||v|| = 1),
    Cosine Similarity = Dot Product đơn thuần.

    Công thức:
        cos_sim(q, d_i) = q · d_i  (vì ||q|| = ||d_i|| = 1)

    Args:
        query_emb: Vector (512,) đã normalize L2.
        db_embs:   Ma trận (N, 512) đã normalize L2.

    Returns:
        np.ndarray (N,) với giá trị trong [-1, 1].
        Giá trị càng gần 1 → càng giống nhau.
    """
    # Dot product: (N, 512) @ (512,) = (N,)
    similarities = db_embs @ query_emb
    return similarities


def identify_face(
    query_embedding: np.ndarray,
    db: FaceDatabase,
    threshold: float = COSINE_THRESHOLD
) -> tuple[str, float]:
    """
    Nhận diện danh tính khuôn mặt bằng Cosine Similarity.

    Quy trình:
      1. Normalize L2 query embedding (nếu chưa normalize).
      2. Tính batch cosine similarity với toàn bộ database.
      3. Lấy điểm cao nhất → nếu > threshold → nhận diện được.

    Args:
        query_embedding: Vector (512,) từ frame hiện tại.
        db:              FaceDatabase đã load.
        threshold:       Ngưỡng [0.0, 1.0] để chấp nhận nhận diện.

    Returns:
        Tuple (label, score):
          - label: Tên người hoặc "Unknown"
          - score: Điểm similarity [0.0, 1.0]
    """
    if db.is_empty:
        return "Unknown", 0.0

    # Normalize L2 query vector
    norm = np.linalg.norm(query_embedding)
    if norm > 1e-6:
        query_emb_normalized = query_embedding / norm
    else:
        return "Unknown", 0.0

    # Tính cosine similarity với toàn bộ database (1 phép nhân ma trận)
    similarities = cosine_similarity_batch(query_emb_normalized, db.embeddings)

    # Lấy index và điểm cao nhất
    best_idx   = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score >= threshold:
        return db.labels[best_idx], best_score
    else:
        return "Unknown", best_score


# ============================================================
# SECTION 3.5: VISUAL LIP MOVEMENT (MAR - Active Speaker Detection)
# ============================================================

import threading


def calculate_mar(face) -> float:
    """
    Tính Mouth Aspect Ratio (MAR) từ facial landmarks.
    Sử dụng 68-point 3D/2D landmarks (face.landmark_3d_68) với các điểm môi 60..67.
    """
    if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
        try:
            lmk = face.landmark_3d_68[:, :2]  # (68, 2)
            # Các điểm mốc khoé môi và môi trong:
            # 60: khoé trái, 64: khoé phải
            # 61, 62, 63: môi trên trong
            # 67, 66, 65: môi dưới trong
            p1_p5 = np.linalg.norm(lmk[60] - lmk[64])
            if p1_p5 > 1e-5:
                p2_p8 = np.linalg.norm(lmk[61] - lmk[67])
                p3_p7 = np.linalg.norm(lmk[62] - lmk[66])
                p4_p6 = np.linalg.norm(lmk[63] - lmk[65])
                mar = (p2_p8 + p3_p7 + p4_p6) / (2.0 * p1_p5)
                return float(mar)
        except Exception:
            pass

    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        try:
            lmk = face.landmark_2d_106
            corner_dist = np.linalg.norm(lmk[52] - lmk[61])
            if corner_dist > 1e-5:
                vertical_dist = np.linalg.norm(lmk[56] - lmk[66])
                return float(vertical_dist / corner_dist)
        except Exception:
            pass

    return 0.0


class MARTracker:
    """
    Quản lý lịch sử MAR của từng khuôn mặt qua các frame (Sliding Window Buffer).
    Xác định trạng thái đang nói (Active Speaker) dựa trên độ biến thiên std(MAR).
    """

    def __init__(self, window_size: int = 15, std_threshold: float = 0.02, min_avg_mar: float = 0.015):
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.min_avg_mar = min_avg_mar
        self.history = {}
        self.lock = threading.Lock()

    def update(self, face_id: str, mar: float) -> tuple[float, bool]:
        """Cập nhật MAR và trả về (mar_std, is_speaking)."""
        with self.lock:
            if face_id not in self.history:
                self.history[face_id] = []

            h = self.history[face_id]
            h.append(mar)
            if len(h) > self.window_size:
                h.pop(0)

            if len(h) < 4:
                return 0.0, False

            mar_std = float(np.std(h))
            avg_mar = float(np.mean(h))

            # Người đang nói có MAR biến thiên đủ cao VÀ trung bình MAR không quá phẳng
            is_speaking = (mar_std >= self.std_threshold) and (avg_mar >= self.min_avg_mar)
            return round(mar_std, 4), is_speaking

    def cleanup_old_users(self, active_ids: set[str]):
        """Dọn dẹp lịch sử của các khuôn mặt rời khỏi khung hình."""
        with self.lock:
            for key in list(self.history.keys()):
                if key not in active_ids:
                    del self.history[key]


mar_tracker = MARTracker()


def process_frame(
    frame: np.ndarray,
    app: FaceAnalysis,
    db: FaceDatabase,
    threshold: float = COSINE_THRESHOLD
) -> tuple[np.ndarray, list]:
    """
    Xử lý 1 frame từ camera: detect → embed → MAR lip movement → nhận diện → vẽ kết quả.
    Returns:
        tuple (output_frame, faces)
    """
    output_frame = frame.copy()

    faces = app.get(output_frame)
    active_ids = set()

    for face in faces:
        embedding = face.embedding
        label, score = identify_face(embedding, db, threshold)

        # Tính MAR và theo dõi biến thiên cử động môi
        mar = calculate_mar(face)

        # Key định danh cho tracker
        if label != "Unknown":
            face_key = label
        else:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            face_key = f"unknown_{int((x1+x2)/2)//60}_{int((y1+y2)/2)//60}"

        active_ids.add(face_key)
        mar_std, is_speaking = mar_tracker.update(face_key, mar)

        # Gán thuộc tính bổ sung trực tiếp vào face object
        face.label = label
        face.score = score
        face.mar = round(mar, 3)
        face.mar_std = mar_std
        face.is_speaking = is_speaking

        # Vẽ kết quả lên frame
        _draw_face_annotation(output_frame, face.bbox, label, score, mar=mar, is_speaking=is_speaking)

    mar_tracker.cleanup_old_users(active_ids)

    # Hiển thị số lượng khuôn mặt phát hiện được
    cv2.putText(
        output_frame,
        f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output_frame, faces


def _draw_face_annotation(
    frame: np.ndarray,
    bbox: np.ndarray,
    label: str,
    score: float,
    mar: float = 0.0,
    is_speaking: bool = False
):
    """
    Hàm nội bộ: Vẽ bounding box, nhãn và trạng thái người đang nói (Speaking 🗣️).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Màu box:
    # Nếu đang nói (Speaking) -> Vàng chanh/Cyan viền dày (0, 255, 255)
    # Nhận diện được -> Xanh lá (0, 220, 50)
    # Unknown -> Đỏ (0, 50, 220)
    if is_speaking:
        color = (0, 255, 255)  # Yellow / Cyan highlight
        thickness = 3
    elif label != "Unknown":
        color = (0, 220, 50)
        thickness = 2
    else:
        color = (0, 50, 220)
        thickness = 2

    # Vẽ bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Text hiển thị: "Tên (0.87)" hoặc "Tên [SPEAKING]"
    speaking_tag = " [SPEAKING]" if is_speaking else ""
    display_text = f"{label} ({score:.2f}){speaking_tag}"

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thick = 2
    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, font_thick)

    label_y = max(y1 - 10, text_h + 10)
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - baseline - 4),
        (x1 + text_w + 4, label_y + 4),
        color,
        cv2.FILLED
    )

    # Nếu đang nói, chữ màu đen trên nền vàng nổi bật; ngược lại chữ trắng
    text_color = (0, 0, 0) if is_speaking else (255, 255, 255)

    cv2.putText(
        frame,
        display_text,
        (x1 + 2, label_y),
        font,
        font_scale,
        text_color,
        font_thick,
        cv2.LINE_AA
    )


# ============================================================
# SECTION 5: VÒNG LẶP CHÍNH (Main Loop)
# ============================================================

def main_loop(app: FaceAnalysis, db: FaceDatabase):
    """
    Vòng lặp chính: Đọc frame từ webcam và nhận diện theo thời gian thực.

    Phím điều khiển:
      - [Q] hoặc [ESC]: Thoát
      - [S]:            Chụp và lưu screenshot

    Args:
        app: FaceAnalysis đã khởi tạo.
        db:  FaceDatabase đã load vào RAM.
    """
    log.info(f"Đang mở webcam (index={CAMERA_INDEX}) ...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error(f"Không thể mở webcam index={CAMERA_INDEX}!")
        return

    # Thiết lập độ phân giải camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer lag

    # Lấy FPS thực tế của camera
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    log.info(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
             f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {actual_fps:.0f}fps")

    log.info("Nhận diện bắt đầu! Nhấn [Q] hoặc [ESC] để thoát, [S] để chụp màn hình.")

    # Biến đo FPS
    fps_counter  = 0
    fps_display  = 0.0
    fps_timer    = time.perf_counter()

    screenshot_dir = Path(__file__).parent / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Không đọc được frame từ camera, thử lại...")
            time.sleep(0.1)
            continue

        # --- Xử lý nhận diện ---
        start_t = time.perf_counter()
        annotated_frame, faces = process_frame(frame, app, db)
        inference_ms = (time.perf_counter() - start_t) * 1000

        # --- Tính và hiển thị FPS ---
        fps_counter += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.perf_counter()

        # Overlay thông tin FPS và inference time
        _draw_overlay_info(annotated_frame, fps_display, inference_ms, db)

        # --- Hiển thị ---
        cv2.imshow("Face Recognition - InsightFace + ONNX Runtime", annotated_frame)

        # --- Xử lý phím ---
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q hoặc ESC
            log.info("Người dùng yêu cầu thoát.")
            break
        elif key in (ord('s'), ord('S')):    # S = Screenshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_path = screenshot_dir / f"capture_{ts}.jpg"
            cv2.imwrite(str(save_path), annotated_frame)
            log.info(f"Screenshot đã lưu: {save_path}")

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    log.info("Đã giải phóng camera và đóng cửa sổ.")


def _draw_overlay_info(
    frame: np.ndarray,
    fps: float,
    inference_ms: float,
    db: FaceDatabase
):
    """
    Hàm nội bộ: Vẽ thông tin hệ thống lên góc dưới-trái frame.

    Hiển thị:
      - FPS: Frames per second của pipeline tổng thể
      - Inference: Thời gian xử lý mỗi frame (ms)
      - DB: Số người trong database

    Args:
        frame:        Frame BGR để vẽ (in-place).
        fps:          FPS hiện tại.
        inference_ms: Thời gian inference (ms).
        db:           FaceDatabase để lấy số người.
    """
    h, w = frame.shape[:2]
    overlay_lines = [
        f"FPS: {fps:.1f}",
        f"Inference: {inference_ms:.1f}ms",
        f"DB: {len(db.labels)} nguoi",
        f"Threshold: {COSINE_THRESHOLD}",
        f"Model: {MODEL_NAME}",
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1
    line_h     = 22
    padding    = 8
    start_y    = h - padding - (len(overlay_lines) * line_h)

    # Nền tối cho overlay
    overlay_bg_h = len(overlay_lines) * line_h + padding * 2
    overlay_bg_w = 200
    bg_roi = frame[start_y - padding : h, 0 : overlay_bg_w]
    bg_roi[:] = (bg_roi * 0.4).astype(np.uint8)  # Làm tối 60%

    for i, line in enumerate(overlay_lines):
        y_pos = start_y + i * line_h
        cv2.putText(
            frame, line,
            (padding, y_pos),
            font, font_scale,
            (180, 230, 180),  # Màu xanh nhạt
            font_thick,
            cv2.LINE_AA
        )


# ============================================================
# SECTION 6: ENTRY POINT
# ============================================================

def main():
    """
    Hàm chính: Điều phối toàn bộ pipeline.

    Thứ tự thực thi:
      1. Khởi tạo model (ONNX Runtime tối ưu)
      2. Load database vào RAM
      3. Bắt đầu vòng lặp nhận diện thời gian thực
    """
    log.info("=" * 60)
    log.info("  FACE RECOGNITION PIPELINE")
    log.info(f"  Model: {MODEL_NAME} | Threads: {INTRA_OP_NUM_THREADS}")
    log.info(f"  Threshold: {COSINE_THRESHOLD} | DB: {DATABASE_DIR}")
    log.info("=" * 60)

    # Bước 1: Khởi tạo model InsightFace với ONNX Runtime tối ưu
    app = init_model()

    # Bước 2: Load database vào RAM (chỉ chạy 1 lần lúc startup)
    db = load_database(app)

    if db.is_empty:
        log.warning(
            "Database chưa có dữ liệu!\n"
            f"Hãy thêm ảnh vào: {DATABASE_DIR}\n"
            "Cấu trúc: database/<tên_người>/<ảnh.jpg>\n"
            "Chương trình vẫn chạy nhưng chỉ hiển thị 'Unknown'."
        )

    # Bước 3: Bắt đầu nhận diện thời gian thực
    main_loop(app, db)

    log.info("Pipeline đã kết thúc.")


if __name__ == "__main__":
    # Tối ưu bổ sung: Giới hạn OpenMP threads (tránh over-subscription)
    # Đặt trước khi import bất kỳ thư viện nào sử dụng OpenMP/MKL
    os.environ.setdefault("OMP_NUM_THREADS",  str(INTRA_OP_NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS",  str(INTRA_OP_NUM_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(INTRA_OP_NUM_THREADS))

    main()
