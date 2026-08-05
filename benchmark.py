"""
============================================================
  benchmark.py
  Đo hiệu suất pipeline: so sánh cấu hình ONNX Runtime khác nhau.
  Chạy: python benchmark.py
============================================================
"""

import time
import logging
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def benchmark_cosine_similarity(n_embeddings: int = 1000, n_iterations: int = 10000):
    """
    Đo tốc độ tính Cosine Similarity cho database với N người.
    Kiểm tra xem phép nhân ma trận (batch) có nhanh hơn vòng lặp không.
    """
    log.info(f"\n{'='*50}")
    log.info(f"BENCHMARK: Cosine Similarity | DB={n_embeddings} vectors | {n_iterations} iterations")

    # Tạo dữ liệu giả
    db_embeddings = np.random.randn(n_embeddings, 512).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    db_embeddings = db_embeddings / norms

    query = np.random.randn(512).astype(np.float32)
    query = query / np.linalg.norm(query)

    # --- Phương pháp 1: Batch (dot product ma trận) ---
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        sims = db_embeddings @ query
        best = np.argmax(sims)
    batch_time = (time.perf_counter() - t0) / n_iterations * 1000

    # --- Phương pháp 2: Vòng lặp Python (để so sánh) ---
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        sims = [np.dot(db_embeddings[i], query) for i in range(n_embeddings)]
        best = np.argmax(sims)
    loop_time = (time.perf_counter() - t0) / n_iterations * 1000

    log.info(f"  Batch  (matrix @):   {batch_time:.4f} ms/query")
    log.info(f"  Loop   (python for): {loop_time:.4f} ms/query")
    log.info(f"  Speedup:             {loop_time / batch_time:.1f}x")


def check_onnx_providers():
    """Hiển thị các ONNX Runtime execution providers khả dụng."""
    log.info(f"\n{'='*50}")
    log.info("ONNX Runtime Execution Providers:")
    providers = ort.get_available_providers()
    for p in providers:
        log.info(f"  - {p}")
    log.info(f"ONNX Runtime version: {ort.__version__}")


def check_thread_config():
    """Kiểm tra cấu hình thread hiện tại."""
    log.info(f"\n{'='*50}")
    log.info("Thread Configuration Test:")

    for n_threads in [1, 2, 4, 8]:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        log.info(f"  SessionOptions(intra_op_threads={n_threads}) ✓ tạo thành công")


if __name__ == "__main__":
    check_onnx_providers()
    check_thread_config()
    benchmark_cosine_similarity(n_embeddings=100,  n_iterations=50000)
    benchmark_cosine_similarity(n_embeddings=1000, n_iterations=10000)
    log.info("\nBenchmark hoàn tất!")
