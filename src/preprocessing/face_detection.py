import os
import cv2
from tqdm import tqdm
from retinaface import RetinaFace


def detect_and_crop_faces(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    videos = os.listdir(input_dir)

    for video in tqdm(videos, desc=f"Processing {input_dir}"):
        video_path = os.path.join(input_dir, video)
        output_video_path = os.path.join(output_dir, video)

        os.makedirs(output_video_path, exist_ok=True)

        frames = os.listdir(video_path)

        for frame_file in frames:
            frame_path = os.path.join(video_path, frame_file)
            img = cv2.imread(frame_path)

            if img is None:
                continue

            try:
                detections = RetinaFace.detect_faces(img)
            except:
                continue

            if isinstance(detections, dict) and len(detections) > 0:
                # pick largest face (better than random)
                best_face = None
                max_area = 0

                for key in detections:
                    x1, y1, x2, y2 = detections[key]["facial_area"]
                    area = (x2 - x1) * (y2 - y1)

                    if area > max_area:
                        max_area = area
                        best_face = (x1, y1, x2, y2)

                if best_face is not None:
                    x1, y1, x2, y2 = best_face

                    face = img[y1:y2, x1:x2]

                    if face.size == 0:
                        continue

                    face = cv2.resize(face, (224, 224))

                    cv2.imwrite(os.path.join(output_video_path, frame_file), face)


def process_dataset(base_input, base_output):
    classes = ["real", "fake"]

    for cls in classes:
        input_dir = os.path.join(base_input, cls)
        output_dir = os.path.join(base_output, cls)

        os.makedirs(output_dir, exist_ok=True)
        detect_and_crop_faces(input_dir, output_dir)


if __name__ == "__main__":
    input_dir = "data/frames"
    output_dir = "data/faces"

    process_dataset(input_dir, output_dir)
