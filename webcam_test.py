import cv2
import torch
import numpy as np
import os
from sam2.sam.build_sam import build_sam2_video_predictor

# --- CONFIGURATION ---
OUTPUT_VIDEO = "webcam_recording.mp4"
CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
CONFIG = "sam2_hiera_t.yaml"
# Use CUDA if available, fallback to MPS (Mac) or CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def record_video(filename):
    """
    Opens webcam, records to filename.
    Controls: 'r' to start recording, 'q' to stop/save.
    """
    cap = cv2.VideoCapture(0)  # 0 usually usually default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MacOS friendly codec
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    recording = False
    print("\n--- WEBCAM MODE ---")
    print("Press 'r' to START recording.")
    print("Press 'q' to STOP and save.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Visual feedback
        display_frame = frame.copy()
        if recording:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)  # Red dot
            cv2.putText(
                display_frame,
                "REC",
                (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            out.write(frame)
        else:
            cv2.putText(
                display_frame,
                "Press 'r' to record",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Webcam - Press Q to Quit", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            recording = True
            print("Recording started...")
        elif key == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return os.path.exists(filename)


def get_click_from_video(video_path):
    """
    Plays video. User pauses with SPACE, clicks to select point.
    Returns: frame_index, [x, y]
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    selected_point = None
    paused = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv2.EVENT_LBUTTONDOWN and paused:
            selected_point = (x, y)
            print(f"Selected Point: {selected_point}")

    cv2.namedWindow("Select Object")
    cv2.setMouseCallback("Select Object", mouse_callback)

    print("\n--- SELECTION MODE ---")
    print("1. Press SPACE to pause on the object.")
    print("2. CLICK on the object.")
    print("3. Press ENTER to confirm and track.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                frame_idx = 0
                continue
            frame_idx += 1
            current_frame = frame
        else:
            # Show the frame nicely paused
            display = current_frame.copy()
            if selected_point:
                # Draw green crosshair
                x, y = selected_point
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    f"Selected: {x},{y}",
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Select Object", display)

        if not paused:
            cv2.imshow("Select Object", current_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 32:  # SPACE
            paused = not paused
        elif key == 13:  # ENTER
            if selected_point:
                break
            else:
                print("Please click a point first!")
        elif key == 27:  # ESC
            return None, None

    cap.release()
    cv2.destroyAllWindows()
    # Note: frame_idx in loop is 1-based usually after read, adjust to 0-based
    return max(0, frame_idx - 1), list(selected_point)


def main():
    # 1. Record
    if not record_video(OUTPUT_VIDEO):
        print("No video recorded.")
        return

    # 2. Select
    start_frame, click_coords = get_click_from_video(OUTPUT_VIDEO)
    if click_coords is None:
        print("Selection cancelled.")
        return

    print(f"\nTarget locked! Frame: {start_frame}, Coords: {click_coords}")

    # 3. Load SAM 2
    print("Loading SAM 2...")
    model = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=DEVICE)

    inference_state = model.init_state(video_path=OUTPUT_VIDEO)

    # 4. Inject Prompt
    print("Initializing tracker...")

    # --- FIX: REMOVED MANUAL SCALING ---
    # SAM 2 handles the scaling internally based on inference_state["video_height"]
    # We just pass the raw coordinates from the webcam/video.

    print(f"Original Click: {click_coords}")

    # Pass the RAW coordinates directly
    points = np.array([click_coords], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    model.add_new_points(
        inference_state=inference_state,
        frame_idx=start_frame,
        obj_id=1,
        points=points,
        labels=labels,
    )

    # 5. Track & Visualize
    print("Tracking... (Check the 'Tracking Result' window)")

    # FIX 1: Use correct argument name 'start_frame_idx'
    video_stream = model.propagate_in_video(
        inference_state, start_frame_idx=start_frame
    )

    print("Tracking complete. Preparing visualization...")

    cap = cv2.VideoCapture(OUTPUT_VIDEO)

    # Robust Seeking
    print(f"Seeking to frame {start_frame}...")
    current_frame_index = 0
    while current_frame_index < start_frame:
        ret, _ = cap.read()
        if not ret:
            print(f"Error: Could not seek to frame {start_frame}. Video ended.")
            return
        current_frame_index += 1

    print("Starting playback loop...")

    cv2.namedWindow("Tracking Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking Result", 800, 600)

    # FIX 2: Correctly iterate through the stream
    for frame_idx, obj_ids, video_res_masks in video_stream:
        # Read the corresponding video frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame {frame_idx} from video file.")
            break

        print(f"Displaying frame {frame_idx}...")

        # FIX 3: Simplify Mask Processing
        # video_res_masks is already [Num_Objs, 1, H, W] and resized to video resolution
        # We take the first object (index 0) and the first channel (index 0)
        mask_tensor = video_res_masks[0, 0]  # -> Shape: [H, W]

        # Threshold: SAM output is logits, so > 0.0 means object presence
        mask_binary = (mask_tensor > 0.0).cpu().numpy().astype(np.uint8)

        # Create colored overlay (Red)
        colored_mask = np.zeros_like(frame)
        colored_mask[:, :, 2] = 255  # Red channel

        # Apply overlay
        alpha = 0.5
        mask_indices = mask_binary > 0
        frame[mask_indices] = cv2.addWeighted(
            frame[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0
        )

        cv2.imshow("Tracking Result", frame)

        # Increase wait time to 100ms to make it clearly visible
        if cv2.waitKey(100) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
