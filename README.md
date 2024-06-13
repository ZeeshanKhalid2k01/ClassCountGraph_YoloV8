# Vehicle Detection and Count Graph

This project uses YOLOv8 for vehicle detection and tracking in a video file. It updates the count graph of detected vehicle classes every 30 frames by default. You can change this interval by modifying the value of the `plot_update_interval` variable.

## Usage

1. **Load YOLO Model**

   Ensure you have the YOLOv8 model file (`yolov8s.pt`) in your working directory or update the path accordingly:

   ```python
   model = YOLO('yolov8s.pt')
   ```
   
2. **Open Video File for Reading**

   Make sure the video file (traffic.mp4) is available in your working directory or update the path accordingly:
   
   ```python
   video_path = "traffic.mp4"
   ```

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install opencv-python
pip install pandas
pip install numpy
pip install ultralytics
pip install matplotlib
```

## Note
This code updates the count graph after every 30 frames by default. If you want to change the interval, you can change the value of the (plot_update_interval) variable.
   

