from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
import cv2
import os
from PIL import Image

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('last.pt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/vidpred', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            video_path = os.path.join('static', 'uploaded_video.mp4')
            file.save(video_path)
            
            return redirect(url_for('video_feed', video_path=video_path))
    
    return render_template('index.html')

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

    cap.release()

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path', None)
    
    if video_path:
        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return 'Error: No video file provided.'

@app.route('/imgpred', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded image to a temporary location
            image_path = "static/uploaded_image.jpg"
            file.save(image_path)

            # Run inference on the uploaded image
            results = model(image_path)  # results list

            # Visualize the results
            for i, r in enumerate(results):
                # Plot results image
                im_bgr = r.plot()  # BGR-order numpy array
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

                # Save the result image
                result_image_path = "static/result_image.jpg"
                im_rgb.save(result_image_path)

            # Remove the uploaded image
            os.remove(image_path)

            # Render the HTML template with the result image path
            return render_template('index.html', image_path=result_image_path)

    # If no file is uploaded or GET request, render the form
    return render_template('index.html', image_path=None)

@app.route('/live_feed')
def live_feed():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_frames():
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam

    while True:
        success, frame = cap.read()

        if success:
            # Perform prediction on the frame using your YOLO model
            results = model(frame)
            annotated_frame = results[0].plot()

            # Convert the annotated frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame bytes as part of the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
