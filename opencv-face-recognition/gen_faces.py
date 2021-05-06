import cv2

video=cv2.VideoCapture(0)
str_path='./dataset/jaehyeon/'
cnt=0
while True:
    ret,frame=video.read()
    if ret is False:
        break
    cv2.imshow('face',frame)
    cv2.imwrite(str_path+'{}.jpg'.format(cnt),frame)
    cnt+=1

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()