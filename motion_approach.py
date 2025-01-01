import cv2
import os

def motion_based_frame_extraction(video_path, output_folder, motion_threshold, max_frames=20):
    """
    Extract frames based on motion intensity.
    
    Args:
        video_path (str): Path to the input video.
        output_folder (str): Folder to save the selected frames.
        motion_threshold (float): Threshold for detecting motion between frames.
        max_frames (int): Maximum number of frames to select.
    
    Returns:
        list: List of saved frame file paths.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    motion_frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            # Calculate motion intensity (difference between frames)
            diff = cv2.absdiff(prev_frame, gray)
            motion_score = diff.mean()
            if motion_score > motion_threshold:
                motion_frames.append(frame_idx)

        prev_frame = gray
        frame_idx += 1

    cap.release()

    # Limit the number of frames to max_frames
    if len(motion_frames) > max_frames:
        interval = len(motion_frames) // max_frames
        motion_frames = motion_frames[::interval][:max_frames]

    # Save selected frames
    saved_files = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    file_index = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or file_index > len(motion_frames):
            break

        if frame_idx in motion_frames:
            output_path = os.path.join(output_folder, f"{file_index}.jpg")
            cv2.imwrite(output_path, frame)
            saved_files.append(output_path)
            file_index += 1

        frame_idx += 1

    cap.release()
    return saved_files

# Example usage
video_path = "./test_video2.mp4"
output_folder = "motion_selected_frames_testvid2"
motion_threshold = 15  # Adjust based on video characteristics
selected_frames = motion_based_frame_extraction(video_path, output_folder, motion_threshold)    #Here set max_frames=25 when doing for test video 3

print(f"Motion-Based Frames Saved at: {selected_frames}")
