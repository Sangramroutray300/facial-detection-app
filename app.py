from flask import Flask, render_template, Response
import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]
            emotion = result['dominant_emotion']
            region = result["region"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            # Draw bounding box and emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36,255,12), 2)

            # Save to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame([[timestamp, emotion]], columns=["timestamp", "emotion"])
            df.to_csv("emotion_log.csv", mode='a', header=False, index=False)

        except Exception as e:
            print("DeepFace Error:", e)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
