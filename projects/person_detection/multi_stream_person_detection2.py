import os
import subprocess
import cv2
import time
import threading

# Folder path where your videos are
VIDEO_FOLDER = "./"

# Step 1: Convert all .webm files (with AV1) to .mp4 (H264)
def convert_webm_to_mp4(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-strict", "-2",
        output_path,
    ]
    print(f"Converting {input_path} to {output_path}...")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error converting {input_path}:\n{result.stderr.decode()}")
    else:
        print(f"Conversion finished: {output_path}")

# Step 2: Person detection for each video stream
def person_detection(video_path, stream_id):
    print(f"Starting detection on {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Load your detection model here, e.g. OpenVINO, Tensorflow, etc.
    # For demo, using OpenCV's pre-trained HOG + SVM person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Detect people
        boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8,8))

        # Draw bounding boxes
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show FPS on window title
        frame_count += 1
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = end_time
            print(f"Stream {stream_id}: FPS = {fps:.2f}")

        cv2.imshow(f"Stream {stream_id}", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection finished on {video_path}")

def main():
    # 1. Convert all .webm files to .mp4
    files = os.listdir(VIDEO_FOLDER)
    webm_files = [f for f in files if f.endswith(".webm")]
    converted_files = []

    for webm_file in webm_files:
        base_name = os.path.splitext(webm_file)[0]
        mp4_file = base_name + "_converted.mp4"
        input_path = os.path.join(VIDEO_FOLDER, webm_file)
        output_path = os.path.join(VIDEO_FOLDER, mp4_file)

        # Convert only if converted file doesn't exist
        if not os.path.exists(output_path):
            convert_webm_to_mp4(input_path, output_path)
        else:
            print(f"Converted file already exists: {mp4_file}")

        converted_files.append(output_path)

    # 2. Run detection on all converted mp4 videos using multithreading
    threads = []
    for i, video_path in enumerate(converted_files):
        t = threading.Thread(target=person_detection, args=(video_path, i+1))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()

