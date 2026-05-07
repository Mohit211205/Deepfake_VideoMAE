import os
import cv2
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, num_frames=16):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    current_frame = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_indices:
            frame_name = f"frame_{saved_count:03d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

        current_frame += 1

    cap.release()


def process_class(input_dir, output_dir, num_frames=16):
    videos = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov"))]

    for video_file in tqdm(videos, desc=f"Processing {input_dir}"):
        video_path = os.path.join(input_dir, video_file)

        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)

        extract_frames_from_video(video_path, video_output_dir, num_frames)


def process_dataset(base_input, base_output, num_frames=16):
    classes = ["real", "fake"]

    for cls in classes:
        input_dir = os.path.join(base_input, cls)
        output_dir = os.path.join(base_output, cls)

        os.makedirs(output_dir, exist_ok=True)
        process_class(input_dir, output_dir, num_frames)


if __name__ == "__main__":
    input_dir = "data/raw_videos"
    output_dir = "data/frames"

    process_dataset(input_dir, output_dir, num_frames=16)
