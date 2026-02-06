import cv2
from flask import Flask, Response, request
from ultralytics import YOLO

app = Flask(__name__)

# ================= LOAD MODELS =================
face_model = YOLO("models/yolov8s-face-lindevs.pt")
object_model = YOLO("models/yolov8s-oiv7.pt")

camera = cv2.VideoCapture(0)

# ================= FRAME GENERATOR =================
def generate_frames(video):
    while True:
        success, frame = video.read()
        if not success:
            break

        frame = cv2.resize(frame, (1000, 560))

        # -------- FACE DETECTION (PERSON) --------
        face_results = face_model(frame, conf=0.5)
        for r in face_results:
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Person | {conf*100:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # -------- OBJECT DETECTION --------
        object_results = object_model(frame, conf=0.5)
        for r in object_results:
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(
                    frame,
                    f"Object | {conf*100:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 165, 0),
                    2
                )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

# ================= ROUTES =================
@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>
<title>AI Vision Dashboard</title>

<style>
body {
    margin: 0;
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
    text-align: center;
}

h1 {
    margin-top: 30px;
    font-size: 42px;
}

.subtitle {
    opacity: 0.8;
    margin-bottom: 40px;
}

.dashboard {
    display: flex;
    justify-content: center;
    gap: 50px;
    margin-top: 40px;
}

.card {
    background: #020617;
    width: 320px;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 0 30px rgba(34,197,94,0.25);
    transition: transform 0.3s;
}

.card:hover {
    transform: scale(1.05);
}

.icon {
    font-size: 80px;
    margin-bottom: 20px;
}

.btn {
    display: inline-block;
    margin-top: 20px;
    padding: 15px 28px;
    background: #22c55e;
    color: black;
    font-weight: bold;
    font-size: 18px;
    text-decoration: none;
    border-radius: 12px;
    border: none;
    cursor: pointer;
}

.btn:hover {
    background: #16a34a;
}

input {
    margin-top: 15px;
    color: white;
}

footer {
    margin-top: 60px;
    opacity: 0.6;
    font-size: 14px;
}
</style>
</head>

<body>
    <h1>🚀 AI Vision System</h1>
    <p class="subtitle">Face → Person | Objects → Object | Real-Time Detection</p>

    <div class="dashboard">

        <div class="card">
            <div class="icon">🎥</div>
            <h2>Live Webcam</h2>
            <p>Real-time face & object detection</p>
            <a class="btn" href="/webcam" target="_blank">Start Camera</a>
        </div>

        <div class="card">
            <div class="icon">📂</div>
            <h2>Upload Video</h2>
            <p>Analyze recorded video files</p>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="video" required><br>
                <button class="btn" type="submit">Upload & Detect</button>
            </form>
        </div>

    </div>

    <footer>
        AI Vision System | WIDER FACE + Open Images | Flask + YOLOv8
    </footer>
</body>
</html>
"""

@app.route("/webcam")
def webcam():
    return Response(
        generate_frames(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]
    path = "uploaded_video.mp4"
    file.save(path)
    video = cv2.VideoCapture(path)

    return Response(
        generate_frames(video),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
