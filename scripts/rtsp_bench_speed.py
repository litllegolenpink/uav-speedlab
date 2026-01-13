# scripts/rtsp_bench_speed.py
# -*- coding: utf-8 -*-
import os, sys, time, csv, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import psutil

from src.pipeline.speed_pipeline import SpeedPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rtsp", type=str, default="rtsp://localhost:8554/mystream")
    p.add_argument("--config", type=str, default="configs/speed_config.yaml")
    p.add_argument("--out_csv", type=str, default="perf_log_with_model.csv")
    p.add_argument("--out_jsonl", type=str, default="tracks.jsonl")
    p.add_argument("--target_fps", type=float, default=3.0)   # 抽帧频率（每秒处理几帧）
    p.add_argument("--max_frames", type=int, default=300)
    p.add_argument("--rtsp_transport", type=str, default="tcp", choices=["tcp", "udp"])
    p.add_argument("--reconnect_sleep", type=float, default=0.5)
    return p.parse_args()


def open_rtsp(rtsp_url: str, rtsp_transport: str):
    if rtsp_transport == "tcp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    return cap


def main():
    args = parse_args()

    # 初始化 pipeline（只做一次）
    pipe = SpeedPipeline(args.config)

    # 打开 RTSP
    cap = open_rtsp(args.rtsp, args.rtsp_transport)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RTSP: {args.rtsp}")

    interval = 1.0 / max(args.target_fps, 1e-6)
    next_t = time.monotonic()

    proc = psutil.Process(os.getpid())
    proc.cpu_percent(None)  # warm-up，避免第一行 CPU=0 偏差

    with open(args.out_csv, "w", newline="") as fcsv, open(args.out_jsonl, "w") as fjs:
        w = csv.writer(fcsv)
        w.writerow(["frame", "infer_ms", "cpu_percent", "ram_mb"])

        i = 0
        while i < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                # RTSP 断流/读不到帧 -> 重连
                cap.release()
                time.sleep(args.reconnect_sleep)
                cap = open_rtsp(args.rtsp, args.rtsp_transport)
                continue

            now = time.monotonic()
            if now < next_t:
                continue
            next_t = now + interval

            t0 = time.perf_counter()

            # === 关键：跑“完整流水线”单帧 step() ===
            # fps_hint 用 target_fps（抽帧后的处理频率），让 dt 更贴近实际
            vis, js = pipe.step(frame, fps_hint=args.target_fps)

            infer_ms = (time.perf_counter() - t0) * 1000.0

            cpu = proc.cpu_percent(None)
            ram_mb = proc.memory_info().rss / (1024 * 1024)

            w.writerow([i, round(infer_ms, 3), round(cpu, 1), round(ram_mb, 1)])
            fjs.write(json.dumps(js, ensure_ascii=False) + "\n")

            i += 1

    cap.release()
    print(f"[OK] wrote {args.out_csv} and {args.out_jsonl}")


if __name__ == "__main__":
    main()
