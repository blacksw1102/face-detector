import face_recognition

image = face_recognition.load_image_file("./image/Elon Musk.jpg")
face_locations = face_recognition.face_locations(image)

print('location : ', face_locations)

# face_locations is now an array listing the co-ordinates of each face!