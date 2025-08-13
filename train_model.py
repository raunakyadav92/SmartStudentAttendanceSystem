# train_model.py
import os
import cv2
import numpy as np

DATASET_PATH = "dataset"
TRAINER_PATH = "trainer"
TRAINER_FILE = os.path.join(TRAINER_PATH, "trainer.yml")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_images_and_labels(dataset_path):
    face_samples = []
    ids = []
    label_map = {}  # name -> numeric id
    curr_id = 0

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        # assign id
        label_map[curr_id] = person_name
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200,200))
            face_samples.append(img)
            ids.append(curr_id)
        curr_id += 1

    return face_samples, np.array(ids), label_map

if __name__ == "__main__":
    ensure_dir(TRAINER_PATH)
    faces, ids, label_map = get_images_and_labels(DATASET_PATH)
    if len(faces) == 0:
        print("No faces found. Capture dataset first.")
        exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, ids)
    recognizer.write(TRAINER_FILE)
    # save label map
    import json
    with open(os.path.join(TRAINER_PATH, "labels.json"), "w") as f:
        json.dump(label_map, f)
    print(f"Training complete. Model saved to {TRAINER_FILE}")
    print(f"Label map saved to {os.path.join(TRAINER_PATH, 'labels.json')}")
