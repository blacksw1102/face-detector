import cv2
import face_detector as fd

# 0번 웹캠 인스턴스 바인딩
video_capture = cv2.VideoCapture(0)

while True:
    # frame 추출
    ret, frame = video_capture.read()

    # 이미지 가공
    processedFrame = fd.getFrame(frame)

    # 이미지 출력
    cv2.imshow('Video', frame)

    # 'q' 키 누르면 프로그램 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 릴리즈
video_capture.release()
cv2.destroyAllWindows()