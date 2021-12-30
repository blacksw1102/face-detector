import face_recognition
import cv2
import numpy as np

# 얼굴 이미지 불러오기 (DB 저장 필요)
elon_image = face_recognition.load_image_file(".\image\Elon Musk.jpg")
messi_image = face_recognition.load_image_file(".\image\Messi.jpg")
bezos_image = face_recognition.load_image_file(".\image\Jeff Bezos.jpg")
suwan_image = face_recognition.load_image_file(".\image\Suwan2.jpg")

# 얼굴 피처 값 추출 (DB 저장 필요)
elon_encoding = face_recognition.face_encodings(elon_image)[0]
messi_encoding = face_recognition.face_encodings(messi_image)[0]
bezos_encoding = face_recognition.face_encodings(bezos_image)[0]
suwan_encoding = face_recognition.face_encodings(suwan_image)[0]

# known face encoding value 목록 (DB 저장 필요)
known_face_encodings = [
    elon_encoding,
    suwan_encoding,
    messi_encoding,
    bezos_encoding
]

# known face Name 값 목록 (DB 저장 필요)
known_face_names = [
    "Elon Musk",
    "Suwan",
    "Messi",
    "Jeff Bezos",
]

def getFrame(frame):

    # 실시간 인식된 얼굴 관련 파라미터 초기화
    face_locations = []
    face_encodings = []
    face_names = []

    # 인지 처리 성능 향상을 위해 프레임 사이즈 1/4로 리사이징
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # OpenCV 적용 위해 BGR -> RGB 컨버트
    rgb_small_frame = small_frame[:, :, ::-1]

    # 얼굴 인식 & 얼굴 인지 처리
    face_locations = face_recognition.face_locations(img=rgb_small_frame, number_of_times_to_upsample=2)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        print('matches : ', matches)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)




    # 얼굴 측정 위치 기준 rectangle 색칠
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        print('top:', top, ",right:", right, "bottom:", bottom, "left:", left)

        # 박스 드로잉
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 이름 드로잉
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)