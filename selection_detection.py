import cv2
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import AdaptiveDetector
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import numpy as np
import time


def process_video_with_object_detection(video_path, output_folder, model_ckpt, motion_threshold=15, max_frames=20, detection_threshold=0.4):
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Start timing for frame selection
    start_frame_selection = time.time()

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(min_scene_len=3, window_width=4))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scenes = scene_manager.get_scene_list()
    keyframe_indices = [scene[0].get_frames() for scene in scenes]

    # Limit the number of frames to max_frames
    if len(keyframe_indices) > max_frames:
        interval = len(keyframe_indices) // max_frames
        keyframe_indices = keyframe_indices[::interval][:max_frames]

    

    # End timing for frame selection
    end_frame_selection = time.time()
    frame_selection_time = end_frame_selection - start_frame_selection
    print(f"Frame Selection Time: {frame_selection_time:.2f} seconds")

    # Start timing for object detection
    start_object_detection = time.time()

    #Load Hugging Face object detection model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = AutoModelForObjectDetection.from_pretrained(model_ckpt).to(device)

    # Save selected frames with bounding boxes
    output_files = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    file_index = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or file_index > len(keyframe_indices):
            break

        if frame_idx in keyframe_indices:
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform object detection
            with torch.no_grad():
                inputs = image_processor(images=[image], return_tensors="pt")
                outputs = model(**inputs.to(device))
                target_sizes = torch.tensor([[image.size[1], image.size[0]]]).to(device)
                results = image_processor.post_process_object_detection(
                    outputs, threshold=detection_threshold, target_sizes=target_sizes
                )[0]

            # Draw bounding boxes on the frame
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                score = score.item()
                label = label.item()
                box = [int(i) for i in box]  # Convert box coordinates to integers
                label_name = model.config.id2label[label]

                # Assign a random color to the class
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                # Draw bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

                # Add label and score
                label_text = f"{label_name}: {score:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the frame with bounding boxes
            output_path = os.path.join(output_folder, f"{file_index}.jpg")
            cv2.imwrite(output_path, frame)
            output_files.append(output_path)
            file_index += 1

        frame_idx += 1

    cap.release()

    # End timing for object detection
    end_object_detection = time.time()
    object_detection_time = end_object_detection - start_object_detection
    print(f"Object Detection Time: {object_detection_time:.2f} seconds")

    return output_files


# Example usage
video_path = "./test_video3.mp4"
output_folder = "processed_frames_testvid3"
model_ckpt = "yainage90/fashion-object-detection"
processed_frames = process_video_with_object_detection(video_path, output_folder, model_ckpt, max_frames = 25)

print(f"Frames with Bounding Boxes Saved at: {processed_frames}")
