import cv2
import os

DATA_DIR = '../new_data'
NUM_CLASSES = 30
DATASET_SIZE = 250
CAPTURE_KEY = 'c'
QUIT_KEY = 'q'

lst = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
       'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'ILY', 'PEWPEW', 'Space', 'Hello']

i = 0


def create_directory(dir_path):
    """Vytvorí priečinok, ak neexistuje."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_frame(frame, class_dir, counter):
    """Uloženie snímky do priečinka."""
    cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)


def capture_frame(cap, collecting):
    """Zachytávanie snímky z kamery."""
    ret, frame = cap.read()
    if not collecting:
        cv2.putText(frame, f"To capture {lst[i]} press: " + CAPTURE_KEY + " / q to quit", (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    create_directory(DATA_DIR)

    global i

    for j in range(NUM_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(j))
        create_directory(class_dir)
        print(f'Collecting data for class {j}')

        collecting = False
        while True:
            frame = capture_frame(cap, collecting)
            if cv2.waitKey(25) == ord(CAPTURE_KEY):
                print(f'Starting data collection for class {j}')
                collecting = True
                i += 1
                break
            elif cv2.waitKey(25) == ord(QUIT_KEY):
                cap.release()
                cv2.destroyAllWindows()
                break

        for counter in range(DATASET_SIZE):
            frame = capture_frame(cap, collecting)
            cv2.waitKey(25)
            save_frame(frame, class_dir, counter)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
