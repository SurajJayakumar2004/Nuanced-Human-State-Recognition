import cv2
import pyaudio
import numpy as np

def test_hardware():
    print("--- RHNS v1.0 Hardware Check ---")
    
    # 1. Test Camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[SUCCESS] Camera detected.")
        ret, frame = cap.read()
        if ret:
            print(f"[SUCCESS] Captured frame at resolution: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
    else:
        print("[FAILED] Camera not found.")

    # 2. Test Microphone
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        print("[SUCCESS] Microphone detected and stream opened at 16kHz.")
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"[FAILED] Microphone error: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    test_hardware()