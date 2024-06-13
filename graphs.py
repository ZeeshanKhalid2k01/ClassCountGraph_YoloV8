import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import matplotlib.pyplot as plt
from io import BytesIO

# Load YOLO model
model = YOLO('yolov8s.pt')

# Open the video file for reading
video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Read class labels from a file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize tracker
tracker = Tracker()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1020, 500))

# Initialize list to store the counts of each class over time
history = {cls: [] for cls in class_list if cls in ['car', 'truck', 'bus', 'bicycle', 'motorcycle']}

# Update the plot every n frames
plot_update_interval = 30
frame_count = 0

def create_overlay(history):
    # Create and overlay the line chart
    fig, ax = plt.subplots(figsize=(12, 12))
    for cls, counts in history.items():
        ax.plot(counts, label=cls, linewidth=4.5)
    ax.set_ylabel('Count', fontsize=30)
    ax.set_title('Detected Vehicle Classes Over Time', fontsize=30)
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=35, colors='red')

    # Set transparency
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Save the figure to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)

    # Load the image from the buffer
    plot_img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    plot_img = cv2.imdecode(plot_img, cv2.IMREAD_UNCHANGED)
    plt.close(fig)
    
    return plot_img

plot_img = None

# Main loop to read frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Filter out car objects and count class occurrences
    car_list = []
    class_count = {cls: 0 for cls in history}
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = row
        class_name = class_list[int(d)]
        if class_name in class_count:
            class_count[class_name] += 1
            car_list.append([int(x1), int(y1), int(x2), int(y2)])
            print("Vehicle detected:", class_name)

    # Update tracker with car bounding boxes
    bbox_idx = tracker.update(car_list)
    for bbox in bbox_idx:
        x1, y1, x2, y2, id1 = bbox

        # Generate a unique color for each object based on its ID
        color = (id1 * 50 % 255, id1 * 100 % 255, id1 * 150 % 255)

        # Calculate centroid of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        start_point = (center_x, center_y)

        # Draw arrows from centroid to each corner of the bounding box
        end_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for end_point in end_points:
            cv2.arrowedLine(frame, (int(start_point[0]), int(start_point[1])),
                            (int(end_point[0]), int(end_point[1])), color, 2, tipLength=0.1)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Append current class counts to history
    for cls in class_count:
        history[cls].append(class_count[cls])

    frame_count += 1

    # Update the plot every plot_update_interval frames
    if frame_count % plot_update_interval == 0:
        plot_img = create_overlay(history)

    if plot_img is not None:
        # Resize plot image to fit on frame and overlay it
        plot_img = cv2.resize(plot_img, (600, 400))

        # Extract the alpha channel and create a mask
        alpha_channel = plot_img[:, :, 3]
        rgb_channels = plot_img[:, :, :3]
        mask = alpha_channel == 0

        # Make mask 3 channels
        mask = np.stack([mask] * 3, axis=-1)

        # Overlay the plot image onto the frame
        overlay_start_x = 350
        overlay_start_y = 10
        overlay_end_x = overlay_start_x + plot_img.shape[1]
        overlay_end_y = overlay_start_y + plot_img.shape[0]

        frame[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x][mask] = frame[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x][mask]
        frame[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x][~mask] = rgb_channels[~mask]

    # Write the frame to the output video
    out.write(frame)

    # Display frame
    cv2.imshow("RGB", frame)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()



# NOTE:
#  This code will update the count graph, after every new 30 frames, if you want to change the interval, you can change the value of plot_update_interval variable.