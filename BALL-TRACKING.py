import cv2
import numpy as np

video_path = "C:/Users/Shenbaga Balaji/Downloads/bumrah_skills.mp4"

cap = cv2.VideoCapture(video_path)

# HSV range refined for white ball (you can adjust if needed)
lower_ball = np.array([0, 0, 180])
upper_ball = np.array([180, 40, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Mask for white
    mask = cv2.inRange(hsv, lower_ball, upper_ball)

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius < 8 or radius > 40:
            continue

        x_, y_, w_, h_ = cv2.boundingRect(c)
        aspect_ratio = float(w_) / h_
        if aspect_ratio < 0.85 or aspect_ratio > 1.15:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        break  # Stop after first valid white ball found

    cv2.imshow("White Ball Tracking", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
