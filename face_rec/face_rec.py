import face_recognition
import cv2 as cv
from database_faces import image_db as db

id_device = 0
font = cv.FONT_HERSHEY_SIMPLEX

try:
    webcam = cv.VideoCapture(id_device)

    if not webcam.isOpened:
        raise Exception("error with opening webcam, id devive {0}".format(id_device))
        exit(0)
    else:
        while True:
            ret, frame = webcam.read()

            faces = face_recognition.face_locations(frame) # top, right, bottom, left
            faces_encodings = face_recognition.face_encodings(frame)

            if faces:
                image_name = list()
                for person in db.items():
                    match = face_recognition.compare_faces(person[1], faces_encodings)
                    name = "Unknow"
                    for result in match:
                        name = person[0] if result else "Unknow"

                    image_name.append(name)

                for (top, right, bottom, left), face_name in zip(faces, image_name):
                    cv.rectangle(frame, (left, top - 35), (right, bottom), (255, 0, 0), 1)

                    cv.rectangle(frame, (left, bottom - 30), (right, bottom), (255, 0, 0), cv.FILLED)
                    cv.putText(frame, face_name, (left, bottom - 10), font, 1, (0, 0, 0), 2)

            cv.imshow("frame", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
except Exception as err:
    print(f"{err=}")
finally:
    print("success!")

if __name__ == "__main__":
    webcam.release()
    cv.destroyAllWindows()
