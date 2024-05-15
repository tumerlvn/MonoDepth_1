import cv2
import sys


video = cv2.VideoCapture("humans_cars2_slowed.mp4")

ok, frame = video.read()
print(ok)

if not video.isOpened():
    print('Error')
    sys.exit()
else:
    w = 1920
    h = 1080
    frame = cv2.resize(frame, (w, h))

grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prevFrame = grayFrame

frame_count = 0
frame_freq = 100
first_flag = True

video_out = cv2.VideoWriter("res_cars_humans.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (w,h))
end = False

while True:
    for i in range(3):
        ok, frame = video.read()
        frame_count += 1
        if not ok:
            end = True
            break
    
    if end:
        break

    frame = cv2.resize(frame, (w, h))
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(grayFrame, prevFrame)
    tmpCopy = frame.copy()

    if first_flag or frame_count >= frame_freq:
        first_flag = False
        frame_count = 0

        retval, dstImg = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)    

        dstImg = cv2.blur(dstImg, (7,7))
        retval, dstImg = cv2.threshold(dstImg, 40, 255, cv2.THRESH_BINARY)
        dstImg = cv2.blur(dstImg, (7,7))

        contours, hier = cv2.findContours(dstImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 700:
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
                if w_rect > 5 and h_rect > 5:
                    # rectangles.append(Rect(x_rect, y_rect, w_rect, h_rect))
                    r = 3


    video_out.write(tmpCopy)
    prevFrame = grayFrame

    # cv2.imshow("img", tmpCopy)
    # if cv2.waitKey(1) == ord('q'):
    #     break

video_out.release()
# cv2.destroyAllWindows()