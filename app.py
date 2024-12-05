from flask import Flask, Response
import cv2, time, darknet

app = Flask(__name__)

# Konfigurasi YOLOv3
config_path = "/dataset/yolov3.cfg"       # Path ke file konfigurasi
weights_path = "/dataset/backup/yolov3_final.weights" # Path ke file weights
data_path = "/dataset/data.data"     # Path ke file kelas

# RTSP stream URL
rtsp_url = 'rtsp://aidev:masts2024@192.168.120.2/Streaming/channels/201'

network, class_names, class_colors = darknet.load_network(
    config_path,
    data_path,
    weights_path,
    batch_size=1
)
width = darknet.network_width(network)
height = darknet.network_height(network)

def convert_bbox(detection, width, height):
    """Konversi bbox dari koordinat relatif ke koordinat pixel."""
    x, y, w, h = detection
    x1 = int((x - w / 2) * width)
    y1 = int((y - h / 2) * height)
    x2 = int((x + w / 2) * width)
    y2 = int((y + h / 2) * height)
    return x1, y1, x2, y2

def generate_frames():
    # # Load YOLOv3 model
    # network, class_names, class_colors = darknet.load_network(
    #     config_path,
    #     data_path,
    #     weights_path,
    #     batch_size=1
    # )

    # width = darknet.network_width(network)
    # height = darknet.network_height(network)

    print("Starting video stream...")
    
    # Konfigurasi OpenCV DNN untuk menggunakan GPU jika tersedia
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Inisialisasi RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS: {}".format(fps))
    
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        # prev_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of stream or unable to read frames.")
            break

        # Resize frame agar sesuai dengan input model YOLOv3
        resized_frame = cv2.resize(frame, (width, height))
        darknet_frame = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_frame, resized_frame.tobytes())

        # Inferensi YOLOv3
        detections = darknet.detect_image(network, class_names, darknet_frame, thresh=0.25)
        darknet.free_image(darknet_frame)

        # FPS per detik
        # fps = 1 / (time.time() - prev_time)
        
        # if detections:
        #     print("Detections:")
        #     for label, confidence, bbox in detections:
        #         x1, y1, x2, y2 = convert_bbox(bbox, frame.shape[1], frame.shape[0])
        #         print(" - {}: {}% at [{}, {}, {}, {}]".format(label, confidence, x1, y1, x2, y2))
        # else:
        #     print("No objects detected in this frame.")

        # Tampilkan hasil deteksi
        for label, confidence, bbox in detections:
            x1, y1, x2, y2 = convert_bbox(bbox, frame.shape[1], frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "{} ({}%)".format(label, confidence), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error encoding frame.")
            break

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Route untuk streaming video
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Route utama
@app.route("/")
def index():
    return '''
    <h1>Video Streaming</h1>
    <img src='/video_feed' width="640"/>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
