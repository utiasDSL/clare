import cv2

def show_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return

    print(f"🎥 Showing live feed from /dev/video{camera_id} — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        cv2.imshow(f'Camera {camera_id} - Press q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change this to 0, 2, or 4 depending on which camera you want
    show_camera(camera_id=6)
