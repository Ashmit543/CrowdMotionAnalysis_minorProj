import cv2
import numpy as np

def initialize_video(video_path):
    """Initialize video capture and first frame."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        exit()

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Cannot read the first frame.")
        exit()

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_mask = np.zeros_like(frame)
    hsv_mask[..., 1] = 255  # Set saturation to maximum

    return cap, prev_gray, hsv_mask, frame.shape

def compute_dense_optical_flow(prev_gray, gray, hsv_mask):
    """Compute Dense Optical Flow using Farneback method."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set hue and value for visualization
    hsv_mask[..., 0] = angle * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

def process_video(video_path, output_width=1280, output_height=720):
    """Process video with side-by-side comparison of input and output with adjustable GUI dimensions."""
    cap, prev_gray, hsv_mask, frame_size = initialize_video(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_rgb = compute_dense_optical_flow(prev_gray, gray, hsv_mask)

        # Determine the target size for each video (half of the output width)
        half_width = output_width // 2
        target_size = (half_width, output_height)

        # Resize both frames to fit within the defined GUI window
        frame_resized = cv2.resize(frame, target_size)
        flow_resized = cv2.resize(flow_rgb, target_size)

        # Combine input and output side by side
        combined_display = np.hstack((frame_resized, flow_resized))

        # Show the comparison video in the resized window
        cv2.imshow('Input Video (Left) | Optical Flow Output (Right)', combined_display)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 27 is ESC key
            break

        prev_gray = gray  # Update previous frame

    cap.release()
    cv2.destroyAllWindows()

# Run the function with your video path and adjustable dimensions
video_path = r"D:\Sem 6\Minor Project\Practice_Projects\VideoData\048766426-india-festival-huge-crowd-fest.mp4"
process_video(video_path, output_width=1680, output_height=650)
