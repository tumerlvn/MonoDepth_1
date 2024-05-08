import cv2
import numpy as np

cap = cv2.VideoCapture("/home/tamerlan/Masters/thesis/datasets/traffic_surveilance1.mp4")

backgroundObject = cv2.createBackgroundSubtractorMOG2(history=2)
kernel = np.ones((3,3), np.uint8)
kernel2 = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    fgmask = backgroundObject.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, kernel, iterations=2)
    fgmask = cv2.dilate(fgmask, kernel2, iterations=2)


    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 700:
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy, (x,y), (x + width, y + height), (0, 0, 255), 2)

    cv2.imshow("frameCopy", frameCopy)
    # cv2.imshow("fgmask", fgmask)
    # cv2.imshow("img", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()