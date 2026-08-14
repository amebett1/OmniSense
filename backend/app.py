"""
============================================================
  OmniSense Backend API
  Flask server kết nối giao diện web với InsightFace pipeline
============================================================

Endpoints:
  GET  /api/status              → Trạng thái model & database
  GET  /api/users               → Danh sách người đã đăng ký
  POST /api/users               → Đăng ký khuôn mặt mới
  PUT  /api/users/<id>          → Chỉnh sửa thông tin
  DELETE /api/users/<id>        → Xoá người đã đăng ký
  GET  /api/settings            → Lấy cài đặt hiện tại
  POST /api/settings            → Cập nhật cài đặt
  POST /api/recognize           → Nhận diện khuôn mặt từ ảnh/frame
  GET  /api/video_feed          → MJPEG stream (camera + nhận diện)
  POST /api/camera/start        → Khởi động camera
  POST /api/camera/stop         → Dừng camera
"""

import os
import sys
import json
import time
import uuid
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Tải cấu hình biến môi trường từ .env
load_dotenv()

# Thêm thư mục gốc dự án vào sys.path để import pipeline
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_recognition_pipeline import (
    init_model,
    load_database,
    process_frame,
    identify_face,
    FaceDatabase,
    SUPPORTED_EXTENSIONS,
)

# ============================================================
# CẤU HÌNH
# ============================================================

DATABASE_DIR = PROJECT_ROOT / "face_recognition" / "database"
METADATA_FILE = PROJECT_ROOT / "face_recognition" / "metadata.json"
SETTINGS_FILE = Path(__file__).parent / "settings.json"
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Đường dẫn Piper TTS & Sound output
PIPER_EXE = PROJECT_ROOT / "piper" / "piper.exe"
PIPER_MODEL = PROJECT_ROOT / "models" / "vi_VN-25hours-medium.onnx"
RESPONSE_WAV_PATH = STATIC_DIR / "response.wav"


DEFAULT_SETTINGS = {
    "model": "buffalo_sc",
    "threshold": 0.40,
    "threads": 8,
    "det_size": 640,
    "camera_index": 0,
    "cuda": False,
    "frame_skip": 2,
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
CORS(app)

# ============================================================
# SETTINGS
# ============================================================


def load_settings() -> dict:
    """Tải cài đặt từ file hoặc trả về mặc định."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                # Merge với default để đảm bảo không thiếu key
                merged = {**DEFAULT_SETTINGS, **saved}
                return merged
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict):
    """Lưu cài đặt ra file."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ============================================================
# GLOBAL STATE
# ============================================================


class AppState:
    """Trạng thái toàn cục của ứng dụng."""

    def __init__(self):
        self.model = None
        self.db = None
        self.settings = load_settings()
        self.camera = None
        self.camera_lock = threading.Lock()
        self.is_camera_running = False
        self.latest_frame = None
        self.latest_annotated = None
        self.recognition_results = []
        self.fps = 0.0
        self.inference_ms = 0.0
        self.model_ready = False
        self._camera_thread = None
        self.cam_init_error = None


state = AppState()


# ============================================================
# METADATA (thông tin người dùng: tên hiển thị, chức vụ, giới tính)
# ============================================================


def load_metadata() -> dict:
    """Tải metadata của tất cả người đã đăng ký."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_metadata(metadata: dict):
    """Lưu metadata ra file."""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ============================================================
# MODEL INITIALIZATION (chạy nền khi khởi động)
# ============================================================


def initialize_model():
    """Khởi tạo model InsightFace và load database (chạy nền)."""
    log.info("Đang khởi tạo model InsightFace...")
    try:
        import face_recognition_pipeline as pipeline

        # Cập nhật cấu hình từ settings
        s = state.settings
        pipeline.MODEL_NAME = s.get("model", "buffalo_sc")
        pipeline.COSINE_THRESHOLD = s.get("threshold", 0.40)
        pipeline.INTRA_OP_NUM_THREADS = s.get("threads", 8)
        pipeline.DET_SIZE = (s.get("det_size", 640), s.get("det_size", 640))
        pipeline.CAMERA_INDEX = s.get("camera_index", 0)

        state.model = init_model()
        state.db = load_database(state.model)
        state.model_ready = True
        log.info(
            f"Model sẵn sàng! Database: {len(state.db.labels)} người"
        )
    except Exception as e:
        log.error(f"Lỗi khởi tạo model: {e}")
        state.model_ready = False


# ============================================================
# CAMERA STREAMING
# ============================================================


def camera_loop(init_event: threading.Event = None):
    """Vòng lặp camera chạy trong thread riêng."""
    import face_recognition_pipeline as pipeline

    cam_index = state.settings.get("camera_index", 0)
    log.info(f"Khởi động camera index={cam_index}")
    state.cam_init_error = None

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened() and os.name == "nt":
        log.warning(f"cv2.VideoCapture({cam_index}) thất bại, thử cv2.CAP_DSHOW...")
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        err_msg = f"Không thể mở camera index={cam_index}. Thiết bị có thể đang bận hoặc bị chiếm quyền."
        log.error(err_msg)
        state.cam_init_error = err_msg
        state.is_camera_running = False
        if init_event:
            init_event.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    with state.camera_lock:
        state.camera = cap

    # Đánh dấu mở camera thành công
    if init_event:
        init_event.set()

    fps_counter = 0
    fps_timer = time.perf_counter()

    frame_count = 0
    cached_faces = []
    cached_results = []

    while state.is_camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1
        skip_interval = max(1, int(state.settings.get("frame_skip", 2)))
        run_ai = (frame_count % skip_interval == 0) or not cached_faces

        start_t = time.perf_counter()
        if state.model_ready and state.model is not None and state.db is not None:
            if run_ai:
                threshold = state.settings.get("threshold", pipeline.COSINE_THRESHOLD)
                annotated, faces = process_frame(frame, state.model, state.db, threshold)
                cached_faces = faces

                # Thu thập kết quả nhận diện từ danh sách faces đã xử lý MAR & Active Speaker
                results = []
                metadata = load_metadata()
                for face in faces:
                    label = getattr(face, "label", "Unknown")
                    score = getattr(face, "score", 0.0)
                    mar = getattr(face, "mar", 0.0)
                    mar_std = getattr(face, "mar_std", 0.0)
                    is_speaking = getattr(face, "is_speaking", False)
                    meta = metadata.get(label, {})
                    results.append(
                        {
                            "label": label,
                            "name": meta.get("name", label),
                            "role": meta.get("role", "khác"),
                            "gender": meta.get("gender", "male"),
                            "score": round(float(score), 3),
                            "mar": mar,
                            "mar_std": mar_std,
                            "is_speaking": is_speaking,
                            "bbox": [int(v) for v in face.bbox],
                        }
                    )
                cached_results = results
                state.recognition_results = results
                state.inference_ms = round((time.perf_counter() - start_t) * 1000, 1)
            else:
                # Fast render từ cache bounding boxes (~0.5ms per frame)
                annotated = frame.copy()
                for face in cached_faces:
                    label = getattr(face, "label", "Unknown")
                    score = getattr(face, "score", 0.0)
                    mar = getattr(face, "mar", 0.0)
                    is_speaking = getattr(face, "is_speaking", False)
                    pipeline._draw_face_annotation(annotated, face.bbox, label, score, mar=mar, is_speaking=is_speaking)

                cv2.putText(
                    annotated,
                    f"Faces: {len(cached_faces)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                state.recognition_results = cached_results
        else:
            annotated = frame.copy()
            state.recognition_results = []
            cached_faces = []
            cached_results = []

        # Tính FPS truyền tải thực tế
        fps_counter += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            state.fps = round(fps_counter / elapsed, 1)
            fps_counter = 0
            fps_timer = time.perf_counter()

        state.latest_frame = frame
        state.latest_annotated = annotated

    # Giải phóng camera
    try:
        cap.release()
    except Exception as e:
        log.warning(f"Lỗi khi release camera: {e}")

    with state.camera_lock:
        state.camera = None

    state.is_camera_running = False
    state.latest_frame = None
    state.latest_annotated = None
    state.recognition_results = []
    state.fps = 0.0
    state.inference_ms = 0.0
    log.info("Camera đã dừng.")


def generate_mjpeg():
    """Generator cho MJPEG stream."""
    while state.is_camera_running:
        if state.latest_annotated is not None:
            ret, buffer = cv2.imencode(
                ".jpg", state.latest_annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        time.sleep(0.015)  # Hỗ trợ tối đa ~60fps stream


# ============================================================
# API ROUTES
# ============================================================

# --- Serve frontend ---
@app.route("/")
def serve_index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(str(WEB_DIR), path)


# --- Status ---
@app.route("/api/status")
def api_status():
    return jsonify(
        {
            "model_ready": state.model_ready,
            "model_name": state.settings.get("model", "buffalo_sc"),
            "database_count": len(state.db.labels) if state.db else 0,
            "camera_running": state.is_camera_running,
            "fps": state.fps,
            "inference_ms": state.inference_ms,
            "settings": state.settings,
        }
    )


# --- Users CRUD ---
@app.route("/api/users", methods=["GET"])
def api_get_users():
    metadata = load_metadata()
    users = []

    # Quét thư mục database
    if DATABASE_DIR.exists():
        for person_dir in sorted(DATABASE_DIR.iterdir()):
            if not person_dir.is_dir():
                continue

            dir_name = person_dir.name
            photos = [
                f.name
                for f in person_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            # Lấy metadata nếu có
            meta = metadata.get(dir_name, {})

            users.append(
                {
                    "id": dir_name,
                    "name": meta.get("name", dir_name),
                    "role": meta.get("role", "student"),
                    "gender": meta.get("gender", "male"),
                    "photo_count": len(photos),
                    "photos": photos,
                    "created_at": meta.get("created_at", ""),
                    "has_photo": len(photos) > 0,
                }
            )

    return jsonify(users)


@app.route("/api/users", methods=["POST"])
def api_create_user():
    """Đăng ký khuôn mặt mới."""
    name = request.form.get("name", "").strip()
    role = request.form.get("role", "student")
    gender = request.form.get("gender", "male")

    if not name:
        return jsonify({"error": "Thiếu họ và tên"}), 400

    # Tạo tên thư mục từ tên (chuyển thành snake_case ASCII-safe)
    dir_name = _sanitize_dirname(name)

    # Đảm bảo không trùng
    person_dir = DATABASE_DIR / dir_name
    counter = 1
    while person_dir.exists():
        person_dir = DATABASE_DIR / f"{dir_name}_{counter}"
        counter += 1
    dir_name = person_dir.name

    person_dir.mkdir(parents=True, exist_ok=True)

    # Lưu ảnh
    photos = request.files.getlist("photos")
    if not photos or all(f.filename == "" for f in photos):
        person_dir.rmdir()
        return jsonify({"error": "Cần ít nhất 1 ảnh"}), 400

    saved_count = 0
    for i, photo in enumerate(photos):
        if photo.filename == "":
            continue
        ext = Path(photo.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        save_path = person_dir / f"photo_{i + 1}{ext}"
        photo.save(str(save_path))
        saved_count += 1

    if saved_count == 0:
        person_dir.rmdir()
        return jsonify({"error": "Không có ảnh hợp lệ"}), 400

    # Lưu metadata
    metadata = load_metadata()
    metadata[dir_name] = {
        "name": name,
        "role": role,
        "gender": gender,
        "created_at": datetime.now().isoformat(),
    }
    save_metadata(metadata)

    # Reload database (nền)
    _reload_database_async()

    log.info(f"Đã đăng ký: {name} ({dir_name}) — {saved_count} ảnh")
    return jsonify(
        {
            "success": True,
            "id": dir_name,
            "name": name,
            "photo_count": saved_count,
        }
    )


@app.route("/api/users/<user_id>", methods=["PUT"])
def api_update_user(user_id):
    """Chỉnh sửa thông tin người đã đăng ký."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Dữ liệu không hợp lệ"}), 400

    metadata = load_metadata()
    meta = metadata.get(user_id, {})

    if "name" in data:
        meta["name"] = data["name"].strip()
    if "role" in data:
        meta["role"] = data["role"]
    if "gender" in data:
        meta["gender"] = data["gender"]

    metadata[user_id] = meta
    save_metadata(metadata)

    log.info(f"Đã cập nhật: {user_id} → {meta}")
    return jsonify({"success": True, "user": {**meta, "id": user_id}})


@app.route("/api/users/<user_id>", methods=["DELETE"])
def api_delete_user(user_id):
    """Xoá người đã đăng ký (xoá cả thư mục ảnh)."""
    person_dir = DATABASE_DIR / user_id

    if person_dir.exists() and person_dir.is_dir():
        shutil.rmtree(str(person_dir))
        log.info(f"Đã xoá thư mục: {person_dir}")

    # Xoá metadata
    metadata = load_metadata()
    if user_id in metadata:
        del metadata[user_id]
        save_metadata(metadata)

    # Reload database
    _reload_database_async()

    return jsonify({"success": True})


@app.route("/api/users/<user_id>/photo/<photo_name>")
def api_get_photo(user_id, photo_name):
    """Trả về ảnh cá nhân."""
    photo_dir = DATABASE_DIR / user_id
    return send_from_directory(str(photo_dir), photo_name)


@app.route("/api/users/<user_id>/photos", methods=["POST"])
def api_add_user_photos(user_id):
    """Thêm ảnh mới cho người dùng đã đăng ký."""
    person_dir = DATABASE_DIR / user_id
    if not person_dir.exists() or not person_dir.is_dir():
        return jsonify({"error": "Người dùng không tồn tại"}), 404

    photos = request.files.getlist("photos")
    if not photos or all(f.filename == "" for f in photos):
        return jsonify({"error": "Không có ảnh được gửi lên"}), 400

    saved_count = 0
    ts = int(time.time() * 1000)
    for i, photo in enumerate(photos):
        if photo.filename == "":
            continue
        ext = Path(photo.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        save_path = person_dir / f"photo_{ts}_{i + 1}{ext}"
        photo.save(str(save_path))
        saved_count += 1

    if saved_count == 0:
        return jsonify({"error": "Không có ảnh hợp lệ"}), 400

    _reload_database_async()

    current_photos = [
        f.name
        for f in person_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    log.info(f"Đã thêm {saved_count} ảnh cho người dùng {user_id}")
    return jsonify({"success": True, "added": saved_count, "photos": current_photos})


@app.route("/api/users/<user_id>/photos/<photo_name>", methods=["DELETE"])
def api_delete_user_photo(user_id, photo_name):
    """Xoá 1 ảnh của người dùng."""
    person_dir = DATABASE_DIR / user_id
    if not person_dir.exists() or not person_dir.is_dir():
        return jsonify({"error": "Người dùng không tồn tại"}), 404

    photo_path = person_dir / photo_name
    if not photo_path.exists() or not photo_path.is_file():
        return jsonify({"error": "Ảnh không tồn tại"}), 404

    photo_path.unlink()
    log.info(f"Đã xoá ảnh {photo_name} của người dùng {user_id}")

    _reload_database_async()

    current_photos = [
        f.name
        for f in person_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return jsonify({"success": True, "photos": current_photos})



# --- Settings ---
@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify(state.settings)


@app.route("/api/settings", methods=["POST"])
def api_update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Dữ liệu không hợp lệ"}), 400

    for key in DEFAULT_SETTINGS:
        if key in data:
            state.settings[key] = data[key]

    save_settings(state.settings)
    log.info(f"Cài đặt đã cập nhật: {state.settings}")

    return jsonify({"success": True, "settings": state.settings})


# --- Recognition ---
@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """Nhận diện khuôn mặt từ ảnh gửi lên."""
    if not state.model_ready:
        return jsonify({"error": "Model chưa sẵn sàng"}), 503

    photo = request.files.get("photo")
    if not photo:
        return jsonify({"error": "Thiếu ảnh"}), 400

    # Đọc ảnh
    file_bytes = np.frombuffer(photo.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Không đọc được ảnh"}), 400

    # Nhận diện
    faces = state.model.get(img)
    threshold = state.settings.get("threshold", 0.40)
    results = []
    metadata = load_metadata()

    for face in faces:
        label, score = identify_face(face.embedding, state.db, threshold)
        meta = metadata.get(label, {})
        results.append(
            {
                "label": label,
                "name": meta.get("name", label),
                "role": meta.get("role", ""),
                "gender": meta.get("gender", ""),
                "score": round(float(score), 3),
                "bbox": [int(v) for v in face.bbox],
            }
        )

    return jsonify({"faces": results, "count": len(results)})


# --- Camera ---
@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    # Nếu thread cũ chưa kết thúc, chờ nó giải phóng tối đa 2s
    if state._camera_thread and state._camera_thread.is_alive():
        state.is_camera_running = False
        state._camera_thread.join(timeout=2.0)
        if state._camera_thread.is_alive():
            return jsonify({"error": "Camera đang giải phóng tài nguyên, vui lòng thử lại sau vài giây"}), 500

    if state.is_camera_running:
        return jsonify({"error": "Camera đang chạy rồi"}), 400

    # Reset state trước khi khởi động
    state.latest_frame = None
    state.latest_annotated = None
    state.recognition_results = []
    state.fps = 0.0
    state.inference_ms = 0.0
    state.cam_init_error = None

    init_event = threading.Event()
    state.is_camera_running = True
    state._camera_thread = threading.Thread(target=camera_loop, args=(init_event,), daemon=True)
    state._camera_thread.start()

    # Đợi tối đa 4.0 giây để camera phần cứng khởi tạo xong
    started_ok = init_event.wait(timeout=4.0)
    if not started_ok or state.cam_init_error or not state.is_camera_running:
        state.is_camera_running = False
        err_msg = state.cam_init_error or "Không thể kết nối thiết bị camera (Timeout)"
        return jsonify({"error": err_msg}), 500

    return jsonify({"success": True})


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    state.is_camera_running = False
    if state._camera_thread and state._camera_thread.is_alive():
        state._camera_thread.join(timeout=3.0)

    state.latest_frame = None
    state.latest_annotated = None
    state.recognition_results = []
    state.fps = 0.0
    state.inference_ms = 0.0
    state._camera_thread = None

    return jsonify({"success": True})


@app.route("/api/camera/status")
def api_camera_status():
    return jsonify(
        {
            "running": state.is_camera_running,
            "fps": state.fps,
            "inference_ms": state.inference_ms,
            "faces": state.recognition_results,
        }
    )


@app.route("/api/video_feed")
def api_video_feed():
    """MJPEG video stream cho camera nhận diện."""
    return Response(
        generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# VOICE ASSISTANT (STT + GROQ LLM + PIPER TTS)
# ============================================================

def get_current_detected_user() -> tuple[str, str, str]:
    """
    Hàm callback lấy thông tin người dùng hiện tại xuất hiện trước camera.
    Ưu tiên 1: Người dùng đang có hành vi nói chuyện (is_speaking == True từ MAR Tracker).
    Ưu tiên 2: Khuôn mặt nhận diện gần/rõ nhất đầu tiên.
    """
    if hasattr(state, "recognition_results") and state.recognition_results:
        # Lựa chọn active speaker đang nói chuyện
        speaking_faces = [
            f for f in state.recognition_results
            if f.get("is_speaking") and f.get("label") != "Unknown"
        ]

        if speaking_faces:
            chosen = speaking_faces[0]
            log.info(f"🗣️ Active Speaker detected via MAR: {chosen.get('name')} (is_speaking=True)")
        else:
            chosen = state.recognition_results[0]

        label = chosen.get("label", "Unknown")
        if label != "Unknown":
            metadata = load_metadata()
            user_meta = metadata.get(label, {})
            name = user_meta.get("name", label)
            role = user_meta.get("role", "khác")
            gender = user_meta.get("gender", "male")
            return name, role, gender

    return "Unknown", "khác", "male"


import wave

_piper_voice = None
_piper_lock = threading.Lock()


def get_piper_voice():
    """Lấy hoặc nạp PiperVoice model vào bộ nhớ RAM (chỉ nạp 1 lần)."""
    global _piper_voice
    if _piper_voice is None:
        with _piper_lock:
            if _piper_voice is None and PIPER_MODEL.exists():
                try:
                    log.info(f"Đang nạp Piper TTS model vào RAM: {PIPER_MODEL}...")
                    from piper import PiperVoice
                    _piper_voice = PiperVoice.load(str(PIPER_MODEL))
                    log.info("✅ Piper TTS model đã được nạp thành công vào RAM!")
                except Exception as e:
                    log.error(f"❌ Lỗi khi nạp PiperVoice: {e}")
    return _piper_voice


def synthesize_piper_tts(text: str, output_path: Path) -> bool:
    """
    Tổng hợp câu trả lời tiếng Việt ra file audio .wav.
    Ưu tiên 1: Dùng PiperVoice Python API (nạp model 1 lần vào RAM, tổng hợp siêu nhanh ~100ms).
    Ưu tiên 2: Fallback qua Subprocess CLI (piper.exe hoặc python -m piper.__main__).
    Ưu tiên 3: Fallback qua gTTS (Google Text-to-Speech) nếu Piper không khả dụng.
    """
    if not text or not text.strip():
        return False

    output_path.parent.mkdir(exist_ok=True, parents=True)

    # 1. Thử dùng PiperVoice Python API (Nhanh & Tối ưu nhất)
    voice = get_piper_voice()
    if voice is not None:
        try:
            with wave.open(str(output_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            log.info(f"✅ Đã tạo file âm thanh TTS (PiperVoice API): {output_path}")
            return True
        except Exception as e:
            log.error(f"❌ Lỗi PiperVoice API: {e}")

    # 2. Fallback Subprocess CLI (nếu có piper.exe hoặc python -m piper.__main__)
    if PIPER_MODEL.exists():
        if PIPER_EXE.exists():
            cmd = [str(PIPER_EXE), "--model", str(PIPER_MODEL), "--output_file", str(output_path)]
        else:
            cmd = [sys.executable, "-m", "piper.__main__", "--model", str(PIPER_MODEL), "--output_file", str(output_path)]

        try:
            process = subprocess.run(
                cmd,
                input=text,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=True
            )
            log.info(f"✅ Đã tạo file âm thanh TTS (Subprocess): {output_path}")
            return True
        except Exception as e:
            log.error(f"❌ Lỗi khi chạy Piper TTS Subprocess: {e}")

    # 3. Fallback gTTS (Google Text-to-Speech online)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='vi')
        mp3_path = output_path.with_suffix('.mp3')
        tts.save(str(mp3_path))
        # Nếu output_path là wav, đổi thành copy / rename mp3 nếu cần, hoặc lưu file wav
        # gTTS xuất mp3, nên nếu output_path là .wav ta save .mp3 rồi đổi lại hoặc để browser chơi
        log.info(f"✅ Đã tạo âm thanh fallback bằng gTTS: {mp3_path}")
        return True
    except Exception as e:
        log.error(f"❌ Lỗi gTTS fallback: {e}")

    return False


@app.route("/static/<path:filename>")
def serve_static_audio(filename):
    """Serve file tĩnh từ thư mục static (ví dụ: response.wav)."""
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    API Trợ lý giọng nói Voice Assistant:
    1. Nhận JSON { "text": user_transcript }.
    2. Lấy (name, role, gender) người dùng hiện tại từ get_current_detected_user().
    3. Tạo System Prompt cá nhân hóa chính xác danh xưng (Thầy/Cô nếu là Giảng viên).
    4. Gọi Groq Cloud LLM (llama-3.3-70b-versatile).
    5. Gọi Piper TTS tổng hợp file static/response.wav.
    """
    data = request.get_json() or {}
    user_transcript = data.get("text", "").strip()

    if not user_transcript:
        return jsonify({"error": "Văn bản đầu vào không được để trống"}), 400

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "Chưa cấu hình GROQ_API_KEY trên Server (file .env)"}), 500

    try:
        from groq import Groq
        groq_client = Groq(api_key=api_key)
    except Exception as e:
        return jsonify({"error": f"Lỗi khởi tạo Groq client: {str(e)}"}), 500

    # 1. Lấy tên, vai trò và giới tính người dùng hiện tại từ camera
    current_name, current_role, current_gender = get_current_detected_user()
    role_norm = str(current_role).lower().strip()
    gender_norm = str(current_gender).lower().strip()
    is_female = gender_norm in ["female", "nữ", "nu", "f"]
    title = "Cô" if is_female else "Thầy"

    log.info(f"🎤 User input: '{user_transcript}' | Detected: '{current_name}' (Role: {current_role}, Gender: {current_gender})")

    # 2. Xây dựng System Prompt linh hoạt theo vai trò & giới tính chuẩn xác (Bảo đảm xưng hô đồng nhất xuyên suốt)
    pronoun_instruction = (
        "QUY TẮC BẮT BUỘC VỀ XƯNG HÔ:\n"
        "- Giữ nguyên xưng hô đồng nhất xuyên suốt TOÀN BỘ cuộc trò chuyện.\n"
    )

    is_lecturer = role_norm in ["giảng_viên", "lecturer", "giang_vien"]
    is_student = role_norm in ["sinh_viên", "student", "sinh_vien"]
    is_grad_or_phd = role_norm in ["học_viên_cao_học", "graduate_student", "hoc_vien_cao_hoc", "nghiên_cứu_sinh", "phd_student", "nghien_cuu_sinh"]

    if current_name and current_name != "Unknown":
        if is_lecturer:
            title = "Cô" if is_female else "Thầy"
            system_prompt = (
                f"Bạn là trợ lý AI lễ phép tại phòng nghiên cứu. "
                f"Bạn đang nói chuyện với Giảng viên {current_name} (Giới tính: {'Nữ' if is_female else 'Nam'}).\n"
                f"{pronoun_instruction}"
                f"- BẮT BUỘC xưng hô là 'Em' và BẮT BUỘC gọi đối phương là '{title} {current_name}'. "
                f"TUYỆT ĐỐI KHÔNG dùng từ 'bạn', KHÔNG nhầm sang '{'Thầy' if is_female else 'Cô'}'.\n"
                "Trả lời ngắn gọn từ 1 đến 3 câu bằng tiếng Việt tự nhiên."
            )
        elif is_grad_or_phd:
            title = "Chị" if is_female else "Anh"
            system_prompt = (
                f"Bạn là trợ lý AI lễ phép tại phòng nghiên cứu. "
                f"Bạn đang nói chuyện với Học viên/Nghiên cứu sinh {current_name} (Giới tính: {'Nữ' if is_female else 'Nam'}).\n"
                f"{pronoun_instruction}"
                f"- BẮT BUỘC xưng hô là 'Em' và BẮT BUỘC gọi đối phương là '{title} {current_name}'. "
                f"TUYỆT ĐỐI KHÔNG dùng từ 'Thầy/Cô', KHÔNG dùng từ 'bạn'.\n"
                "Trả lời ngắn gọn từ 1 đến 3 câu bằng tiếng Việt tự nhiên."
            )
        elif is_student or (not is_lecturer and not is_grad_or_phd):
            system_prompt = (
                f"Bạn là trợ lý AI thân thiện, cởi mở. "
                f"Bạn đang nói chuyện với {current_name if current_name else 'bạn'}.\n"
                f"{pronoun_instruction}"
                f"- Hãy BẮT BUỘC xưng hô là 'Mình' hoặc 'Tôi' và gọi đối phương là tên '{current_name}' hoặc 'bạn'. "
                f"TUYỆT ĐỐI KHÔNG dùng từ 'Thầy/Cô/Anh/Chị', KHÔNG xưng 'Em'.\n"
                "Trả lời ngắn gọn từ 1 đến 3 câu bằng tiếng Việt tự nhiên."
            )
        else:
            system_prompt = (
                f"Bạn là trợ lý AI thân thiện. Bạn đang nói chuyện với {current_name}.\n"
                f"{pronoun_instruction}"
                f"- Hãy BẮT BUỘC xưng hô là 'Mình' hoặc 'Tôi' và gọi đối phương là 'bạn'. "
                f"TUYỆT ĐỐI KHÔNG dùng từ 'Thầy/Cô/Anh/Chị', KHÔNG xưng 'Em'.\n"
                "Trả lời ngắn gọn từ 1 đến 3 câu bằng tiếng Việt tự nhiên."
            )
    else:
        system_prompt = (
            f"Bạn là trợ lý AI thân thiện.\n"
            f"{pronoun_instruction}"
            f"- Hãy BẮT BUỘC xưng hô là 'Mình' hoặc 'Tôi' và gọi đối phương là 'bạn'. "
            f"TUYỆT ĐỐI KHÔNG dùng từ 'Thầy/Cô/Anh/Chị', KHÔNG xưng 'Em'.\n"
            "Trả lời ngắn gọn từ 1 đến 3 câu bằng tiếng Việt tự nhiên."
        )

    # 2.5 Lấy Context từ RAG Database dựa trên lịch sử đàm thoại
    global _chat_history
    rag_context = retrieve_context(user_transcript, history=_chat_history)
    source_tag = "LLM"
    if rag_context:
        source_tag = "RAG"
        system_prompt += (
            f"\n\n[DỮ LIỆU THAM KHẢO RAG TỪ VĂN BẢN CHÍNH THỨC]:\n{rag_context}\n\n"
            f"RÀNG BUỘC BẮT BUỘC KHI TRẢ LỜI RAG:\n"
            f"1. CHÚ Ý TÀI LIỆU SCAN OCR KHÔNG DẤU: Văn bản tham khảo được bóc tách từ file scan nên nhiều từ không có dấu Tiếng Việt (ví dụ: 'tin chi' = 'tín chỉ', 'hoc phan' = 'học phần', 'tich luy' = 'tích lũy', 'diem' = 'điểm'). Bạn HÃY HIỂU CÁC TỪ KHÔNG DẤU NÀY LÀ TIẾNG VIỆT CHUẨN để giải thích chi tiết cho người dùng.\n"
            f"2. Đọc và tổng hợp thông tin từ TẤT CẢ các đoạn trong [DỮ LIỆU THAM KHẢO RAG] ở trên.\n"
            f"3. NẾU trong bất kỳ đoạn nào chứa các từ như 'tin chi', 'hoc phan', 'tich luy', 'giao duc', HÃY TRẢ LỜI TRỰC TIẾP định nghĩa, số lượng và quy định đó cho người dùng.\n"
            f"4. TUYỆT ĐỐI KHÔNG từ chối hoặc trả lời rào trước rằng 'không có thông tin về tín chỉ' khi tài liệu có các từ 'tin chi' / 'hoc phan'.\n"
        )

    # 3. Xây dựng danh sách messages cho Groq Cloud LLM (Bao gồm System Prompt + Lịch sử hội thoại đàm thoại)
    messages = [{"role": "system", "content": system_prompt}]
    
    # Nối tối đa 6 tin nhắn gần đây nhất (3 lượt trao đổi)
    for msg in _chat_history[-6:]:
        messages.append(msg)
        
    messages.append({"role": "user", "content": user_transcript})

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.6,
            max_tokens=300
        )
        raw_reply = completion.choices[0].message.content.strip()
        bot_reply = clean_text_for_tts(raw_reply)
        log.info(f"🤖 Groq LLM Reply: '{bot_reply}'")

        # Cập nhật lịch sử đàm thoại đa lượt (Lưu tối đa 10 tin nhắn gần nhất)
        _chat_history.append({"role": "user", "content": user_transcript})
        _chat_history.append({"role": "assistant", "content": bot_reply})
        if len(_chat_history) > 10:
            _chat_history = _chat_history[-10:]

    except Exception as e:
        log.error(f"❌ Lỗi Groq API: {e}")
        return jsonify({"error": f"Lỗi xử lý LLM: {str(e)}"}), 500

    # 4. Piper TTS
    tts_success = synthesize_piper_tts(bot_reply, RESPONSE_WAV_PATH)

    return jsonify({
        "reply_text": bot_reply,
        "audio_url": "/static/response.wav" if tts_success else None,
        "user_name": current_name,
        "user_role": current_role,
        "user_gender": current_gender,
        "source": source_tag
    })


# ============================================================
# DYNAMIC GREETING SYSTEM (DYNAMIC SYSTEM PROMPT & ROLE-BASED)
# ============================================================

import re

GREETING_WAV_PATH = STATIC_DIR / "greeting.wav"


def clean_text_for_tts(text: str) -> str:
    """
    Chuẩn hóa văn bản cho Piper TTS:
    - Loại bỏ emoji, ghi chú trong ngoặc [hành động] hoặc (hành động).
    - Bỏ định dạng Markdown (*, **, #, `).
    - Làm sạch khoảng trắng dư thừa.
    """
    # Xóa ghi chú hành động dạng [cười], (cười)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    # Xóa ký tự Markdown
    text = re.sub(r"[\*\#\`\_\~]", "", text)

    # Xóa Emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text


_chat_history = []  # Lịch sử đàm thoại đa lượt dạng [{"role": "user"/"assistant", "content": text}]


def retrieve_context(query: str, history: list = None) -> str:
    """
    Truy xuất 2 giai đoạn (Two-Stage Smart Retrieval):
    1. Giai đoạn 1: Tìm kiếm trực tiếp bằng câu hỏi hiện tại (có tự động bổ sung từ khóa mã ngành CN1, CN2... nếu có).
    2. Giai đoạn 2 (Fallback): Nếu câu hỏi độc lập không khớp RAG (do câu quá ngắn/nối tiếp như "cụ thể đi"), mới gộp lịch sử câu hỏi đàm thoại trước đó.
    """
    try:
        from rag_pipeline import get_pipeline
        pipeline = get_pipeline()

        # Tự động mở rộng từ khóa nếu chứa mã ngành ngắn (CN1, CN2, CN12, SAT...)
        search_query = query
        words_upper = [w.upper() for w in re.findall(r'\b[A-Za-z0-9]+\b', query)]
        acronyms = [w for w in words_upper if w.startswith('CN') or w in ['SAT', 'HSA', 'THPT']]
        if acronyms:
            search_query = f"mã ngành {' '.join(acronyms)} {query}"

        # Giai đoạn 1: Thử truy vấn độc lập câu hỏi hiện tại
        context = pipeline.query(search_query)
        if context.strip():
            return context

        # Giai đoạn 2: Fallback gộp câu hỏi lịch sử nếu câu mới đứng độc lập chưa tìm ra kết quả
        recent_user_queries = [
            m["content"] for m in (history or []) if m.get("role") == "user"
        ][-2:]

        if recent_user_queries:
            fallback_query = " ".join(recent_user_queries + [query])
            log.info(f"🔄 Fallback nối lịch sử đàm thoại cho RAG Search: '{fallback_query}'")
            return pipeline.query(fallback_query)

        return ""
    except Exception as e:
        log.error(f"Lỗi truy xuất RAG: {e}")
        return ""


def generate_role_greeting(name: str, role: str, gender: str = "male") -> str:
    """
    Tạo câu chào tự động (Dynamic Greeting System) dựa trên tên, vai trò (role) và giới tính (gender).
    Role hỗ trợ: 'giảng_viên' / 'lecturer', 'sinh_viên' / 'student', 'khác' / 'other'.
    Gender hỗ trợ: 'male' / 'nam', 'female' / 'nữ'.
    """
    api_key = os.getenv("GROQ_API_KEY")
    role_norm = str(role).lower().strip()
    gender_norm = str(gender).lower().strip()
    is_female = gender_norm in ["female", "nữ", "nu", "f"]

    # Phân loại vai trò rõ ràng
    is_lecturer = role_norm in ["giảng_viên", "lecturer", "giang_vien"]
    is_student = role_norm in ["sinh_viên", "student", "sinh_vien"]
    is_grad_student = role_norm in ["học_viên_cao_học", "graduate_student", "hoc_vien_cao_hoc"]
    is_phd_student = role_norm in ["nghiên_cứu_sinh", "phd_student", "nghien_cuu_sinh"]

    title = ("Cô" if is_female else "Thầy") if is_lecturer else ""

    if not api_key:
        log.warning("Chưa cấu hình GROQ_API_KEY trong .env! Trả về câu chào mặc định.")
        if is_lecturer:
            return f"Em chào {title} {name} ạ!"
        elif is_grad_student or is_phd_student:
            pronoun = "Chị" if is_female else "Anh"
            return f"Em chào {pronoun} {name} ạ!"
        elif is_student:
            return f"Chào bạn {name} nhé! Chúc bạn một ngày tốt lành."
        else:
            return "Xin chào bạn, chào mừng bạn đến với phòng lab!"

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except Exception as e:
        log.error(f"Lỗi khởi tạo Groq client: {e}")
        if is_lecturer:
            return f"Em chào {title} {name} ạ!"
        elif is_grad_student or is_phd_student:
            pronoun = "Chị" if is_female else "Anh"
            return f"Em chào {pronoun} {name} ạ!"
        elif is_student:
            return f"Chào bạn {name} nhé!"
        else:
            return f"Xin chào {name}!"

    # Xây dựng System Prompt linh hoạt và NGHIÊM NGẶT theo từng vai trò
    if is_lecturer:
        system_prompt = (
            f"Bạn là trợ lý AI lễ phép và trang trọng tại phòng nghiên cứu. "
            f"Bạn đang gửi câu chào tới Giảng viên {name} (Giới tính: {'Nữ' if is_female else 'Nam'}). "
            f"BẮT BUỘC xưng hô là 'Em' và BẮT BUỘC gọi đối phương chính xác là '{title} {name}'. "
            f"TUYỆT ĐỐI KHÔNG gọi bằng tên trống, KHÔNG dùng từ 'bạn', KHÔNG dùng nhầm sang '{'Thầy' if is_female else 'Cô'}'. "
            "RÀNG BUỘC BẮT BUỘC: Trả lời siêu ngắn gọn từ 1 đến 2 câu, dưới 20 từ. "
            "KHÔNG dùng emoji, KHÔNG dùng markdown (**), KHÔNG ghi chú hành động như [cười]."
        )
        user_prompt = f"Tạo câu chào lễ phép tới Giảng viên {title} {name}."
    elif is_grad_student or is_phd_student:
        pronoun = "Chị" if is_female else "Anh"
        title_name = f"{'Học viên' if is_grad_student else 'Nghiên cứu sinh'} {name}"
        system_prompt = (
            f"Bạn là trợ lý AI lễ phép tại phòng nghiên cứu. "
            f"Bạn đang gửi câu chào tới {title_name} (Giới tính: {'Nữ' if is_female else 'Nam'}). "
            f"BẮT BUỘC xưng hô là 'Em' và gọi đối phương là '{pronoun} {name}'. "
            f"TUYỆT ĐỐI KHÔNG dùng từ 'bạn', KHÔNG dùng 'Thầy' hay 'Cô'. "
            "RÀNG BUỘC BẮT BUỘC: Trả lời siêu ngắn gọn từ 1 đến 2 câu, dưới 20 từ. "
            "KHÔNG dùng emoji, KHÔNG dùng markdown (**), KHÔNG ghi chú hành động như [cười]."
        )
        user_prompt = f"Tạo câu chào lễ phép tới {title_name}."
    elif is_student:
        system_prompt = (
            f"Bạn là trợ lý AI thân thiện, cởi mở như bạn bè tại phòng nghiên cứu. "
            f"Bạn đang gửi câu chào tới bạn Sinh viên tên là '{name}'. "
            f"BẮT BUỘC gọi tên trực tiếp là '{name}' hoặc dùng từ 'bạn'. Xưng hô 'Mình' hoặc 'Tôi'. "
            "TUYỆT ĐỐI KHÔNG dùng từ 'Thầy', KHÔNG dùng từ 'Cô', KHÔNG xưng hô 'Em' đối với sinh viên. "
            "RÀNG BUỘC BẮT BUỘC: Trả lời siêu ngắn gọn từ 1 đến 2 câu, dưới 20 từ. "
            "KHÔNG dùng emoji, KHÔNG dùng markdown (**), KHÔNG ghi chú hành động như [cười]."
        )
        user_prompt = f"Tạo câu chào thân thiện tới bạn sinh viên {name}."
    else:  # "khác" / "other" / "Unknown"
        system_prompt = (
            f"Bạn là trợ lý AI lịch sự và hiếu khách tại phòng nghiên cứu. "
            f"Bạn đang gửi câu chào tới vị khách tên là '{name}'. "
            "Gọi đối phương là 'bạn' hoặc dùng tên trực tiếp. "
            "TUYỆT ĐỐI KHÔNG dùng từ 'Thầy' hay 'Cô'. "
            "RÀNG BUỘC BẮT BUỘC: Trả lời siêu ngắn gọn từ 1 đến 2 câu, dưới 20 từ. "
            "KHÔNG dùng emoji, KHÔNG dùng markdown (**), KHÔNG ghi chú hành động như [cười]."
        )
        user_prompt = f"Tạo câu chào lịch sự tới khách {name}."

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=60
        )
        raw_reply = completion.choices[0].message.content.strip()
        cleaned_reply = clean_text_for_tts(raw_reply)
        log.info(f"✨ Dynamic Greeting ({role_norm}, {name}): '{cleaned_reply}'")
        return cleaned_reply
    except Exception as e:
        log.error(f"❌ Lỗi Groq API khi tạo câu chào: {e}")
        if is_lecturer:
            return f"Em chào {title} {name} ạ!"
        elif is_student:
            return f"Chào bạn {name} nhé!"
        else:
            return "Xin chào bạn, chào mừng bạn đến với phòng lab!"



@app.route("/api/greet", methods=["POST", "GET"])
def api_greet():
    """
    Endpoint sinh câu chào tự động:
    POST / GET payload/params: { "name": "Phong", "role": "giảng_viên", "gender": "male" }
    """
    if request.method == "POST":
        data = request.get_json() or {}
        name = data.get("name")
        role = data.get("role")
        gender = data.get("gender")
    else:
        name = request.args.get("name")
        role = request.args.get("role")
        gender = request.args.get("gender")

    # Nếu không truyền name, lấy tự động từ nhận diện camera hiện tại
    if not name or name == "Unknown":
        if hasattr(state, "recognition_results") and state.recognition_results:
            first_face = state.recognition_results[0]
            label = first_face.get("label", "Unknown")
            if label != "Unknown":
                metadata = load_metadata()
                meta = metadata.get(label, {})
                name = meta.get("name", label)
                role = meta.get("role", "khác")
                gender = meta.get("gender", "male")
            else:
                name = "Unknown"
                role = "khác"
                gender = "male"
        else:
            name = "bạn"
            role = "khác"
            gender = "male"

    if not role:
        role = "khác"
    if not gender:
        gender = "male"

    # 1. Sinh câu chào động
    greeting_text = generate_role_greeting(name, role, gender)

    # 2. Tổng hợp Piper TTS audio
    tts_success = synthesize_piper_tts(greeting_text, GREETING_WAV_PATH)

    return jsonify({
        "name": name,
        "role": role,
        "gender": gender,
        "greeting_text": greeting_text,
        "audio_url": "/static/greeting.wav" if tts_success else None
    })





# ============================================================
# RAG DOCUMENT MANAGEMENT API
# ============================================================

_APP_DIR = Path(__file__).resolve().parent          # backend/
_PROJECT_ROOT = _APP_DIR.parent                     # OmniSense/
RAG_DOCS_DIR = _PROJECT_ROOT / "data" / "rag_docs"
RAG_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(original: str) -> str:
    """Giữ ký tự tiếng Việt, chỉ loại bỏ ký tự nguy hiểm."""
    import unicodedata
    name = unicodedata.normalize("NFC", original)
    # Chỉ xoá ký tự thật sự nguy hiểm cho filesystem
    name = name.replace("..", "_").replace("/", "_").replace("\\", "_")
    name = name.replace("\0", "")
    return name.strip() or "unnamed"


@app.route("/api/documents", methods=["GET"])
def list_documents():
    files = []
    if RAG_DOCS_DIR.exists():
        for filename in os.listdir(RAG_DOCS_DIR):
            if filename.lower().endswith(('.pdf', '.txt', '.doc', '.docx')):
                files.append(filename)
    return jsonify({"documents": files})


@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith(('.pdf', '.txt', '.doc', '.docx')):
        return jsonify({"error": "Invalid file format"}), 400

    filename = _safe_filename(file.filename)
    file_path = RAG_DOCS_DIR / filename
    file.save(str(file_path))

    # Trigger RAG pipeline processing (singleton — no reload)
    try:
        from rag_pipeline import get_pipeline
        pipeline = get_pipeline()
        success = pipeline.process_file(str(file_path))
        if not success:
            log.warning(f"Không trích xuất được text từ {filename}")
    except Exception as e:
        log.error(f"Lỗi xử lý RAG cho {filename}: {e}")
        return jsonify({"error": "Lỗi khi nạp tài liệu vào RAG", "details": str(e)}), 500

    return jsonify({"message": f"Đã tải lên và xử lý {filename} thành công"})


@app.route("/api/documents/<filename>", methods=["DELETE"])
def delete_document(filename):
    file_path = RAG_DOCS_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        from rag_pipeline import get_pipeline
        pipeline = get_pipeline()
        pipeline.remove_file(filename)
    except Exception as e:
        log.error(f"Lỗi xoá vector cho {filename}: {e}")

    try:
        os.remove(str(file_path))
    except OSError as e:
        log.error(f"Lỗi xoá file {filename}: {e}")
        return jsonify({"error": "Lỗi khi xóa tài liệu", "details": str(e)}), 500

    return jsonify({"message": f"Đã xóa {filename}"})



# ============================================================
# HELPERS
# ============================================================


def _sanitize_dirname(name: str) -> str:
    """Chuyển tên thành tên thư mục an toàn."""
    import unicodedata
    import re

    # Bỏ dấu tiếng Việt
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))

    # Thay khoảng trắng bằng underscore, xoá ký tự đặc biệt
    ascii_name = re.sub(r"[^\w\s-]", "", ascii_name)
    ascii_name = re.sub(r"[\s-]+", "_", ascii_name.strip())
    return ascii_name.lower()


def _reload_database_async():
    """Reload database trong thread riêng (không block API)."""

    def _reload():
        if state.model is not None:
            log.info("Đang reload database...")
            state.db = load_database(state.model)
            log.info(f"Database reloaded: {len(state.db.labels)} người")

    thread = threading.Thread(target=_reload, daemon=True)
    thread.start()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Khởi tạo model trong thread nền
    init_thread = threading.Thread(target=initialize_model, daemon=True)
    init_thread.start()

    log.info("=" * 60)
    log.info("  OMNISENSE BACKEND API")
    log.info(f"  Web UI: http://localhost:5000")
    log.info(f"  Database: {DATABASE_DIR}")
    log.info("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
