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
            annotated, faces = process_frame(frame, state.model, state.db, threshold)

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

    # 2. Xây dựng System Prompt linh hoạt theo vai trò & giới tính chuẩn xác
    if current_name and current_name != "Unknown":
        if role_norm in ["giảng_viên", "lecturer", "giang_vien"]:
            system_prompt = (
                f"Bạn là trợ lý AI lễ phép và trang trọng tại phòng nghiên cứu. "
                f"Bạn đang nói chuyện với Giảng viên {current_name} (Giới tính: {'Nữ' if is_female else 'Nam'}). "
                f"BẮT BUỘC xưng hô là 'Em' và BẮT BUỘC gọi đối phương là '{title} {current_name}'. "
                f"TUYỆT ĐỐI KHÔNG dùng từ 'bạn', KHÔNG nhầm sang '{'Thầy' if is_female else 'Cô'}'. "
                "Trả lời ngắn gọn từ 1 đến 2 câu bằng tiếng Việt tự nhiên."
            )
        elif role_norm in ["sinh_viên", "student", "sinh_vien"]:
            system_prompt = (
                f"Bạn là trợ lý AI thân thiện, cởi mở như bạn bè. "
                f"Bạn đang nói chuyện với bạn sinh viên {current_name}. "
                f"Hãy xưng hô là 'Mình' hoặc 'Tôi' và gọi tên '{current_name}' hoặc 'bạn'. "
                "TUYỆT ĐỐI KHÔNG dùng từ 'Thầy', KHÔNG dùng từ 'Cô', KHÔNG xưng 'Em' với sinh viên. "
                "Trả lời ngắn gọn từ 1 đến 2 câu bằng tiếng Việt tự nhiên."
            )
        else:
            system_prompt = (
                f"Bạn là trợ lý AI lịch sự và hiếu khách. Bạn đang nói chuyện với {current_name}. "
                "TUYỆT ĐỐI KHÔNG dùng từ 'Thầy' hay 'Cô'. "
                "Hãy trả lời ngắn gọn từ 1 đến 2 câu bằng tiếng Việt tự nhiên."
            )
    else:
        system_prompt = (
            "Bạn là trợ lý AI lịch sự và hiếu khách. "
            "Bạn đang nói chuyện với một người chưa quen biết. Trả lời ngắn gọn từ 1 đến 2 câu bằng tiếng Việt."
        )

    # 3. Gọi Groq Cloud LLM
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_transcript}
            ],
            temperature=0.6,
            max_tokens=150
        )
        raw_reply = completion.choices[0].message.content.strip()
        bot_reply = clean_text_for_tts(raw_reply)
        log.info(f"🤖 Groq LLM Reply: '{bot_reply}'")

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
        "user_gender": current_gender
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

    title = ("Cô" if is_female else "Thầy") if is_lecturer else ""

    if not api_key:
        log.warning("Chưa cấu hình GROQ_API_KEY trong .env! Trả về câu chào mặc định.")
        if is_lecturer:
            return f"Em chào {title} {name} ạ!"
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
