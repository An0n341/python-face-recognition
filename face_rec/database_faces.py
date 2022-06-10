import face_recognition

image_1 = face_recognition.load_image_file("faces/obama.jpg")
image_2 = face_recognition.load_image_file("faces/elon_musk.jpg")

image1_encodings = face_recognition.face_encodings(image_1)[0]
image2_encodings = face_recognition.face_encodings(image_2)[0]

image_db = {
    "Barack Obama": image1_encodings,
    "Elon Musk": image2_encodings
}
