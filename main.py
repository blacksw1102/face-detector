import cv2
from face_detector import FaceDetector

# 앱단에서 소유하는 유저 데이터
users = [
    {
        "user_id" : 1,
        "name" : "Elon Musk",
        "face_feature" : FaceDetector.getFacefeature(".\image\Elon Musk.jpg")
    },
    {
        "user_id" : 2,
        "name" : "Jeff Bezos",
        "face_feature" : FaceDetector.getFacefeature(".\image\Jeff Bezos.jpg")
    },
    {
        "user_id" : 3,
        "name" : "Messi",
        "face_feature" : FaceDetector.getFacefeature(".\image\Messi.jpg")
    }
]

# 얼굴 인식 정보를 프레임에 드로잉
def drawFaceInfoInFrame(frame, detected_face_info):

    # 얼굴 측정 위치 기준 rectangle 색칠
    for face_info in detected_face_info:

        # 파라미터 초기화
        name = face_info['name']
        top, right, bottom, left = face_info['location']

        # 박스 드로잉
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 이름 드로잉
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# 유저 정보(List<Map>)에서 특정 Key의 Value들을 List로 리턴
def getValuesInUsers(users, key):
    names = []
    for user in users:
        names.append(user[key])
    return names

def main():

    # 웹캠 인스턴스 바인딩
    video_capture = cv2.VideoCapture(0)

    while True:
        # 프레임 추출                                                                 # 앱 영역
        ret, frame = video_capture.read()

        # known 유저정보에서 이름, 인코딩값 추출
        known_face_names = getValuesInUsers(users, "name")
        known_face_encodings = getValuesInUsers(users, "face_feature")

        # 인식된 얼굴 정보 추출                                                        # 서드파티 영역
        detected_face_info = FaceDetector.getDetectedFaceInfoInFrame(frame, known_face_names, known_face_encodings, 2)

        # 프레임 드로잉
        drawFaceInfoInFrame(frame, detected_face_info)

        # 프레임 출력                                                                 # 앱 영역
        cv2.imshow('Video', frame)

        # 'q' 키 누르면 프로그램 종료                                                  # 앱 영역
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠 릴리즈                                                                     # 앱 영역
    video_capture.release()
    cv2.destroyAllWindows()

main()