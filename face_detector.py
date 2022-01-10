import face_recognition
import cv2
import numpy as np

class FaceDetector:
    
    @staticmethod
    def getFacefeature(file_path):
        """
        이미지에서 얼굴 피쳐 값을 추출하여 리턴합니다.

        :param file_path: 얼굴 이미지 파일 경로
        :return: 얼굴 랜드마크 값
        """

        face_image = face_recognition.load_image_file(file_path)
        face_encoding = face_recognition.face_encodings(face_image)[0]
        return face_encoding

    @staticmethod
    def getDetectedFaceInfoInFrame(frame, known_face_names, known_face_encodings, number_of_times_to_upsample=1):
        """
        프레임에서 인식된 얼굴정보를 리스트로 구성하여 리턴합닌다.

        :param frame: 프레임 이미지
        :param known_face_names: DB에 등록되어있는 유저의 이름 리스트,
        :param known_face_encodings: DB에 등록되어있는 유저의 랜드마크 값 리스트,
        :param number_of_times_to_upsample:
        :return: [ { "name": "이름", "location": [ top, right, bottom, left ] }, ... ]
        """

        # 프레임 리사이징 (인지 성능 향상 목적)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # BGR -> RGB 컨버트
        rgb_small_frame = small_frame[:, :, ::-1]

        # 프레임에서 인식된 얼굴의 (위치, 랜드마크, 이름) 추출
        small_face_locations = face_recognition.face_locations(img=rgb_small_frame, number_of_times_to_upsample=number_of_times_to_upsample)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, small_face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = known_face_names[best_match_index] if matches[best_match_index] else "Unknown"
            face_names.append(name)

        # 오리지날 프레임 사이즈에 맞게 face_location 값 원복
        face_locations = []
        for (top, right, bottom, left) in small_face_locations:
            face_location = [top * 4, right * 4, bottom * 4, left * 4]
            face_locations.append(face_location)

        # 추출한 얼굴정보(이름, 위치)값을 리스트로 구성하여 리턴한다.
        face_infos = [];
        for idx in range(len(face_names)):
            face_infos.append({ "name" : face_names[idx], "location" : face_locations[idx] })

        return face_infos
