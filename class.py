import cv2
import face_recognition
import numpy as np
import os

#video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

known_face_encodings = []
known_face_names = []

path="Mates\\"
for filenames in os.listdir(path):
    image = face_recognition.load_image_file(path+filenames)
    image_face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(image_face_encoding)
    known_face_names.append(filenames[7:-4])

"""
ankith_image = face_recognition.load_image_file("img.jpg")
ankith_face_encoding = face_recognition.face_encodings(ankith_image)[0]

suresh_image = face_recognition.load_image_file("suresh.jpg")
suresh_face_encoding = face_recognition.face_encodings(suresh_image)[0]
"""

face_locations = []
face_encodings = []
face_names = []
#process_this_frame = True


frame = cv2.imread("test2.jpeg")
#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
rgb_small_frame = frame[:, :, ::-1]

#face_locations = face_recognition.face_locations(rgb_small_frame)
#print(np.shape(rgb_small_frame))
#print(face_locations)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
face_locations = face_cascade.detectMultiScale(rgb_small_frame,1.1,4)
print(face_locations)

for (x, y, w, h) in face_locations:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


face_loc = []
for (x, y, w, h) in face_locations:
    face_loc.append((y,x+w,y+h,x))
print(face_loc)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_loc)
face_names = []

for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
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

    #face_names.append(str(list(np.reshape(np.asarray(face_distances), (1, np.size(face_distances)))[0]))[1:-1])
    face_names.append(name)
print(face_names)
for (top, right, bottom, left), name in zip(face_loc, face_names):
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #top *= 4
    #right *= 4
    #bottom *= 4
    #left *= 4

    #bottom = top+bottom
    #left = right+left

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, name, (left + 10 , bottom -10 ), font, 0.5, (255, 255, 255), 1)


cv2.imshow("yo",frame)
cv2.waitKey(15000)