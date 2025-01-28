import cv2


video_path = r'C:\Users\mbazi\Downloads\P20221101_Video.mp4'


# Initialize video capture
cap = cv2.VideoCapture(video_path)  # Replace with your video file path

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

trajectory = []  # To track the rodent's path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    fgmask = fgbg.apply(blurred)

    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Ignore small objects
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2  # Centroid
            trajectory.append((cx, cy))

            # Draw bounding box and centroid
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw trajectory
    # for i in range(1, len(trajectory)):
    #     cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

    # Show frame
    cv2.imshow('Rodent Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q'
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
