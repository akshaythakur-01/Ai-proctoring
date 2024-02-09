from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib
import camera
from math import hypot
from mouth_tracking import *
from facial_landmarks_detection import *
from blink_detection import *
from gaze_detection import *

app = Flask(__name__)
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            faceCount, faces = detectFace(frame)
            mouthTrack(faces, frame)
            blinkStatus = isBlinking(faces, frame)
            eyeStatus = gazeDetection(faces, frame)
            if len(blinkStatus) >= 3:
                print(blinkStatus[2] + ' - ' + eyeStatus)
            else:
                print("blinkStatus does not have enough elements.")

            # Display frame with detected features
            for face in faces:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            
            # Convert frame to JPEG format and yield it as a response
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
