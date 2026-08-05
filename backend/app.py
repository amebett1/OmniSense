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
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

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

DEFAULT_SETTINGS = {
    "model": "buffalo_sc",
    "threshold": 0.40,
    "threads": 8,
    "det_size": 640,
    "camera_index": 0,
    "cuda": False,
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


def camera_loop():
    """Vòng lặp camera chạy trong thread riêng."""
    import face_recognition_pipeline as pipeline

    cam_index = state.settings.get("camera_index", 0)
    log.info(f"Khởi động camera index={cam_index}")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        log.error(f"Không thể mở camera index={cam_index}")
        state.is_camera_running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    with state.camera_lock:
        state.camera = cap

    fps_counter = 0
    fps_timer = time.perf_counter()

    while state.is_camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Nhận diện khuôn mặt
        start_t = time.perf_counter()
        if state.model_ready and state.model is not None and state.db is not None:
            threshold = state.settings.get("threshold", pipeline.COSINE_THRESHOLD)
            annotated = process_frame(frame, state.model, state.db, threshold)

            # Thu thập kết quả nhận diện
            faces = state.model.get(frame)
            results = []
            for face in faces:
                label, score = identify_face(
                    face.embedding, state.db, threshold
                )
                results.append(
                    {
                        "label": label,
                        "score": round(float(score), 3),
                        "bbox": [int(v) for v in face.bbox],
                    }
                )
            state.recognition_results = results
        else:
            annotated = frame.copy()
            state.recognition_results = []

        inference_ms = (time.perf_counter() - start_t) * 1000
        state.inference_ms = round(inference_ms, 1)

        # Tính FPS
        fps_counter += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            state.fps = round(fps_counter / elapsed, 1)
            fps_counter = 0
            fps_timer = time.perf_counter()

        state.latest_frame = frame
        state.latest_annotated = annotated

    # Giải phóng camera
    cap.release()
    with state.camera_lock:
        state.camera = None
    log.info("Camera đã dừng.")


def generate_mjpeg():
    """Generator cho MJPEG stream."""
    while state.is_camera_running:
        if state.latest_annotated is not None:
            ret, buffer = cv2.imencode(
                ".jpg", state.latest_annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        time.sleep(0.033)  # ~30fps max


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
    if state.is_camera_running:
        return jsonify({"error": "Camera đang chạy rồi"}), 400

    state.is_camera_running = True
    state._camera_thread = threading.Thread(target=camera_loop, daemon=True)
    state._camera_thread.start()
    return jsonify({"success": True})


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    state.is_camera_running = False
    if state._camera_thread:
        state._camera_thread.join(timeout=3)
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
