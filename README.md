# intel_unnati_1

# ✅ Project Title: Scalable Visual AI Pipeline Using DL Streamer on Intel CPU/GPU

> A real-time, multi-stream AI video analytics pipeline for visual surveillance, leveraging Intel OpenVINO™, DL Streamer, and GStreamer.

---

## 📌 Description

This project builds a **scalable visual AI pipeline** that performs:
- **Real-time person detection**
- **Multi-stream video input processing**
- **Optimized inference using Intel OpenVINO™**
- **Pipeline orchestration using DL Streamer and GStreamer**

It is designed for **urban surveillance**, **mega-events monitoring**, and **infrastructure analytics**.

---

## ⚙️ Features

- 🔍 Multi-stream inference (1 to 6 simultaneous video feeds)
- 🚶‍♂️ Person detection using `person-detection-retail-0013` model
- 📉 Real-time FPS calculation for benchmarking
- 🎥 Video saving or frame display with OpenCV
- ⚡ Optimized performance using Intel CPU or integrated GPU
- 🧪 Scalable benchmarking for visual AI tasks

---

## 🧱 Tech Stack

| Component        | Description                         |
|------------------|-------------------------------------|
| **OpenVINO™**     | Intel toolkit for optimized inference |
| **DL Streamer**   | GStreamer-based framework for video AI |
| **GStreamer**     | Media processing pipeline engine     |
| **OpenCV**        | Frame processing and visualization   |
| **Python**        | Scripting & orchestration            |

---

## 📁 Directory Structure

```
person_detection/
├── run_4_streams.py                # Script to run multi-stream inference
├── multi_stream_person_detection.py
├── screenshots/                    # Saved screenshots from detections
├── intel/                          # Downloaded model files
├── models/                         # Optional model storage path
├── mahakumbh_video.mp4            # Sample input video
├── output_stream_*.mp4            # Output processed videos
```

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/kranthia186/intel_unnati_1.git
cd person_detection
```

### 2. Install Dependencies
```bash
sudo apt install python3-pip
pip3 install openvino-dev opencv-python
```

### 3. Download Model
```bash
omz_downloader --name person-detection-retail-0013 --output_dir ./models
```

### 4. Run the Pipeline
```bash
python3 run_4_streams.py
```

---

## 📊 Performance Metrics

| Streams | Avg FPS (CPU) | Notes                |
|---------|---------------|----------------------|
| 1       |  19 FPS       |                      |
| 2       |  17 FPS       | Slight drop          |
| 3       |  18 FPS       | Sustained inference  |
| 4       |  20 FPS       | Max stress test      |

_

---

## 🙋‍♂️ Author

**Alagandula Kranthi**  
Final Year Student | CSIT | Visual AI Developer  
📧 Email: kranthialagandula@gmail.com
