# capture_dataset.py
import cv2
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_faces(name, num_samples=50, save_dir="dataset"):
    ensure_dir(save_dir)
    person_dir = os.path.join(save_dir, name)
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0
    print("Press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            file_path = os.path.join(person_dir, f"{str(count)}.jpg")
            cv2.imwrite(file_path, face_img)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Saved {count}/{num_samples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Capture Faces", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} face images for {name} in {person_dir}")

if __name__ == "__main__":
    person_name = input("Enter person name (no spaces): ").strip()
    capture_faces(person_name, num_samples=50)
