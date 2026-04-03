import cv2
import time

def main():
    """
    Capture and display live camera feed with frame-per-second (FPS) monitoring.

    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # drop CAP_DSHOW on Linux/macOS
    if not cap.isOpened():
        raise RuntimeError("Couldn't open camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WIN_NAME = "USB Camera (press q to quit)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)   # create once with constant name

    # ----- FPS helpers -------------------------------------------------------
    prev_t  = time.perf_counter()
    fps     = 0.0
    counter = 0
    AVG_OVER = 30                                   # frames to average
    # -------------------------------------------------------------------------

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, exiting.")
            break

        # --- FPS calculation -------------------------------------------------
        counter += 1
        if counter >= AVG_OVER:
            curr_t  = time.perf_counter()
            elapsed = curr_t - prev_t
            fps     = counter / elapsed
            prev_t  = curr_t
            counter = 0
            # Update window title (OpenCV ≥ 4.5.2)
            cv2.setWindowTitle(WIN_NAME, f"{WIN_NAME} – {fps:.1f} FPS")
            # If setWindowTitle is unavailable, comment the line above and
            # uncomment the next two lines to overlay text instead:
            # cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # ---------------------------------------------------------------------

        cv2.imshow(WIN_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
