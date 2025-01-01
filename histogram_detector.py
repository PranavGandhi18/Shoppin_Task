from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import HistogramDetector
import os
import cv2

def keyframe_based_frame_extraction(video_path, output_folder, max_frames=20):
    """
    Extract frames based on keyframe detection using PySceneDetect.
    
    Args:
        video_path (str): Path to the input video.
        output_folder (str): Folder to save the selected frames.
        max_frames (int): Maximum number of frames to select.
    
    Returns:
        list: List of saved frame file paths.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Keyframe extraction using PySceneDetect
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(HistogramDetector(threshold=0.1,bins=256,min_scene_len=3))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scenes = scene_manager.get_scene_list()
    keyframe_indices = [scene[0].get_frames() for scene in scenes]

    # Limit the number of frames to max_frames
    if len(keyframe_indices) > max_frames:
        interval = len(keyframe_indices) // max_frames
        keyframe_indices = keyframe_indices[::interval][:max_frames]

    # Save selected frames
    saved_files = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    file_index = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or file_index > len(keyframe_indices):
            break

        if frame_idx in keyframe_indices:
            output_path = os.path.join(output_folder, f"{file_index}.jpg")
            cv2.imwrite(output_path, frame)
            saved_files.append(output_path)
            file_index += 1

        frame_idx += 1

    cap.release()
    return saved_files

# Example usage
video_path = "./test_video1.mp4"
output_folder = "histogram_selected_frames_testvid1"
selected_frames = keyframe_based_frame_extraction(video_path, output_folder,max_frames=25)   

print(f"Keyframe-Based Frames Saved at: {selected_frames}")
