import cv2
import time
import threading
from openvino.runtime import Core

# Set model path correctly
MODEL_XML = "./models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"

# List your 4 video sources here (files or camera indices)
video_sources = [
    "mahakumbh_video_converted_h264.mp4",
    "mahakumbh_video_converted.mp4",
    "mahakumbh_video_h264.mp4",
    "mahakumbh_video.mp4_converted.mp4"
]

def process_stream(video_path, stream_id):
    core = Core()
    model = core.read_model(model=MODEL_XML)
    compiled_model = core.compile_model(model=model, device_name="CPU")
    infer_request = compiled_model.create_infer_request()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Stream {stream_id}] ERROR: Cannot open video source {video_path}")
        return

    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare input blob (resize and convert)
        input_blob = compiled_model.inputs[0]
        n, c, h, w = input_blob.shape

        resized_frame = cv2.resize(frame, (w, h))
        # Convert HWC to CHW and add batch dimension
        input_frame = resized_frame.transpose((2, 0, 1))[None, ...]

        # Run inference
        infer_request.infer({input_blob.any_name: input_frame})

        # Here you would get the output and do postprocessing (e.g., draw boxes)
        # For simplicity, skipping drawing or output saving

        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time

    cap.release()
    print(f"[Stream {stream_id}] Finished. Approx FPS: {fps:.2f}")

def main():
    threads = []
    for i, src in enumerate(video_sources):
        t = threading.Thread(target=process_stream, args=(src, i+1))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All streams finished.")

if __name__ == "__main__":
    main()

