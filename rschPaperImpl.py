import cv2
import numpy as np

def compute_divergence_curl(flow, max_div=None, max_curl=None):
    """Compute divergence and curl with improved normalization."""
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    magnitude[magnitude < 1e-5] = 1e-5  # Avoid division by near-zero
    u = flow[:, :, 0] / magnitude
    v = flow[:, :, 1] / magnitude

    du_dx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
    du_dy = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
    dv_dx = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3)

    divergence = du_dx + dv_dy
    curl = dv_dx - du_dy

    # Initialize or update maximum values over frames
    if max_div is None:
        max_div = np.max(np.abs(divergence))
    else:
        max_div = max(max_div, np.max(np.abs(divergence)))
    if max_curl is None:
        max_curl = np.max(np.abs(curl))
    else:
        max_curl = max(max_curl, np.max(np.abs(curl)))

    # Normalize to [-1, 1]
    if max_div > 0:
        divergence = divergence / max_div
    if max_curl > 0:
        curl = curl / max_curl

    return divergence, curl, max_div, max_curl

def classify_behavior(divergence, curl, frame, div_history=None, curl_history=None):
    """Classify crowd behavior with temporal smoothing and optimized thresholds."""
    h, w = frame.shape[:2]
    behavior_map = np.zeros((h, w), dtype=np.uint8)
    labels = []

    # Initialize history lists
    if div_history is None:
        div_history = []
    if curl_history is None:
        curl_history = []

    # Append current values and maintain history of 5 frames
    div_history.append(divergence)
    curl_history.append(curl)
    if len(div_history) > 5:
        div_history.pop(0)
        curl_history.pop(0)

    # Compute temporal average
    if div_history:
        divergence_avg = np.mean(div_history, axis=0)
        curl_avg = np.mean(curl_history, axis=0)
    else:
        divergence_avg = divergence
        curl_avg = curl

    div_mean = np.mean(np.abs(divergence_avg))
    div_std = np.std(np.abs(divergence_avg))
    curl_mean = np.mean(np.abs(curl_avg))
    curl_std = np.std(np.abs(curl_avg))

    div_threshold = 0.02  # High divergence for fountainhead/bottleneck
    curl_threshold = 0.01  # High curl for arches
    div_pos_threshold = 0.5  # Positive divergence for fountainhead
    div_neg_threshold = -0.015  # Negative divergence for bottleneck

    block_size = 16
    h_blocks, w_blocks = h // block_size, w // block_size
    divergence_blocks = cv2.resize(divergence_avg, (w_blocks, h_blocks), interpolation=cv2.INTER_AREA)
    curl_blocks = cv2.resize(curl_avg, (w_blocks, h_blocks), interpolation=cv2.INTER_AREA)

    for y_block in range(h_blocks):
        for x_block in range(w_blocks):
            div_val = divergence_blocks[y_block, x_block]
            curl_val = curl_blocks[y_block, x_block]

            y_start, y_end = y_block * block_size, (y_block + 1) * block_size
            x_start, x_end = x_block * block_size, (x_block + 1) * block_size

            if abs(div_val) < div_threshold and abs(curl_val) < curl_threshold:
                behavior_map[y_start:y_end, x_start:x_end] = 1  # Lane
                labels.append("Lane")
            elif div_val > div_pos_threshold:
                behavior_map[y_start:y_end, x_start:x_end] = 4  # Fountainhead
                labels.append("Fountainhead")
            elif div_val < div_neg_threshold:
                behavior_map[y_start:y_end, x_start:x_end] = 5  # Bottleneck
                labels.append("Bottleneck")
            elif abs(curl_val) > curl_threshold:
                if curl_val > 0:
                    behavior_map[y_start:y_end, x_start:x_end] = 2  # Clockwise Arch
                    labels.append("Clockwise Arch")
                else:
                    behavior_map[y_start:y_end, x_start:x_end] = 3  # Counterclockwise Arch
                    labels.append("Counterclockwise Arch")

    dominant_behavior = max(set(labels), key=labels.count) if labels else "Unknown"

    print(f"Adaptive Thresholds - div: {div_threshold:.4f}, curl: {curl_threshold:.4f}, "
          f"pos: {div_pos_threshold:.4f}, neg: {div_neg_threshold:.4f}")
    return behavior_map, dominant_behavior, div_history, curl_history

def compute_dense_optical_flow(prev_gray, gray, hsv_mask):
    """Compute Dense Optical Flow using Farneback method and visualize."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = angle * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

def visualize_results(frame, prev_gray, gray, divergence, curl, behavior_map, dominant_behavior):
    """Visualize original frame and optical flow with behavior overlay."""
    # Compute optical flow visualization
    hsv_mask = np.zeros_like(frame)
    hsv_mask[..., 1] = 255  # Set saturation to maximum
    flow_rgb = compute_dense_optical_flow(prev_gray, gray, hsv_mask)

    # Resize to fixed dimensions
    target_size = (840, 720)  # Half of 1680 width, 720 height
    frame_resized = cv2.resize(frame, target_size)
    flow_resized = cv2.resize(flow_rgb, target_size)

    # Combine side by side
    combined_display = np.hstack((frame_resized, flow_resized))

    # Overlay dominant behavior
    cv2.putText(combined_display, f"Dominant Behavior: {dominant_behavior}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Print mean divergence and curl values
    div_mean = np.mean(divergence)
    curl_mean = np.mean(curl)
    print(f"Divergence Mean: {div_mean:.4f}, Curl Mean: {curl_mean:.4f}")

    return combined_display

def process_video(video_path):
    """Process a single video and display visualizations with dynamic frame size."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Get original video frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define resize factor (e.g., 0.5 for half size)
    resize_factor = 0.5
    target_width = int(width * resize_factor)
    target_height = int(height * resize_factor)

    ret, frame1 = cap.read()
    if not ret:
        print(f"No frames in video: {video_path}")
        return
    frame1 = cv2.resize(frame1, (target_width, target_height))
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    max_div, max_curl = None, None
    div_history, curl_history = None, None
    frame_count = 0
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (target_width, target_height))
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        divergence, curl, max_div, max_curl = compute_divergence_curl(flow, max_div, max_curl)
        behavior_map, dominant_behavior, div_history, curl_history = classify_behavior(divergence, curl, frame2,
                                                                                     div_history, curl_history)

        # Pass divergence and curl to visualize_results
        combined = visualize_results(frame2, prev_gray, gray, divergence, curl, behavior_map, dominant_behavior)

        cv2.imshow("Input Video (Left) | Optical Flow Output (Right)", combined)
        print(f"Frame {frame_count}: Dominant Behavior = {dominant_behavior}")

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        prev_gray = gray
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = r"D:\Sem 6\Minor Project\POND5Data\048785099-india-transportation-crowds-ra.mp4"
    print(f"Processing: {video_path}")
    process_video(video_path)

if __name__ == "__main__":
    main()