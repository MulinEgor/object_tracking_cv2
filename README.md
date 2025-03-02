# Object tracking with cv2

Project for detecting and tracking objects (people and vehicles) using OpenCV and MobileNet neural network.

## Project structure

```
src/
├── detection.py - logic for detections with mobile net
├── main.py - main script for starting tracking
weights/ - folder with weights for mobile net
```

## Project Setup and Launch

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Install dependencies:

```bash
uv sync 
```

3. Launch project:

```bash
uv run src/main.py
```

## Configuration Parameters for Tracking and Detection in src/main.py

- confidence_threshold: Confidence threshold for object detection.
- nms_threshold: Overlap threshold for NMS (Non-Maximum Suppression).
- max_age: Maximum number of frames to track an object.
- min_hits: Minimum number of frames to confirm an object.
- iou_threshold: Intersection over Union threshold.
- objects: List of objects to detect.
- cap_path: Path to video file or camera number.
