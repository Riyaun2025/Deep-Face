from fastapi import FastAPI, HTTPException

from fastapi.responses import StreamingResponse, HTMLResponse

import cv2

import threading

import time

from typing import Optional

from pydantic import BaseModel
 
app = FastAPI()
 
class CameraSettings(BaseModel):

    resolution: Optional[tuple] = (640, 480)

    fps: Optional[int] = 30

    brightness: Optional[int] = None

    contrast: Optional[int] = None
 
class CameraStreamer:

    def __init__(self):

        self.camera = None

        self.latest_frame = None

        self.frame_lock = threading.Lock()

        self.is_running = False

        self.settings = CameraSettings()

        self.thread = None

    def update_settings(self, settings: CameraSettings):

        self.settings = settings

        if self.camera is not None:

            if settings.resolution:

                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0])

                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1])

            if settings.brightness is not None:

                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, settings.brightness)

            if settings.contrast is not None:

                self.camera.set(cv2.CAP_PROP_CONTRAST, settings.contrast)

    def start(self):

        if self.is_running:

            return

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            raise RuntimeError("Could not open camera")

        self.update_settings(self.settings)

        self.is_running = True

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

        self.thread.start()

    def stop(self):

        self.is_running = False

        if self.thread is not None:

            self.thread.join()

        if self.camera is not None:

            self.camera.release()

        self.camera = None

    def _capture_loop(self):

        frame_interval = 1.0 / self.settings.fps

        while self.is_running:

            start_time = time.time()

            success, frame = self.camera.read()

            if not success:

                continue

            # Process frame if needed

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            frame_bytes = buffer.tobytes()

            with self.frame_lock:

                self.latest_frame = frame_bytes

            # Maintain frame rate

            elapsed = time.time() - start_time

            sleep_time = max(0, frame_interval - elapsed)

            time.sleep(sleep_time)

    def get_frame(self):

        with self.frame_lock:

            return self.latest_frame
 
camera_streamer = CameraStreamer()
 
@app.on_event("startup")

def startup_event():

    try:

        camera_streamer.start()

    except Exception as e:

        print(f"Failed to start camera: {e}")
 
@app.on_event("shutdown")

def shutdown_event():

    camera_streamer.stop()
 
def generate_frames():

    while True:

        frame = camera_streamer.get_frame()

        if frame is None:

            continue

        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
@app.get("/video_feed")

async def video_feed():

    return StreamingResponse(

        generate_frames(),

        media_type="multipart/x-mixed-replace; boundary=frame"

    )
 
@app.get("/settings", response_model=CameraSettings)

async def get_settings():

    return camera_streamer.settings
 
@app.put("/settings")

async def update_settings(settings: CameraSettings):

    camera_streamer.update_settings(settings)

    return {"message": "Settings updated"}
 
@app.get("/", response_class=HTMLResponse)

async def index():

    return """
<html>
<head>
<title>Camera Stream</title>
<style>

                body { font-family: Arial, sans-serif; margin: 20px; }

                .container { max-width: 800px; margin: 0 auto; }

                .controls { margin: 20px 0; padding: 10px; background: #f0f0f0; border-radius: 5px; }
</style>
</head>
<body>
<div class="container">
<h1>Camera Streaming</h1>
<img src="/video_feed" width="640">
<div class="controls">
<h3>Camera Controls</h3>
<form id="settingsForm">
<label>Resolution:</label>
<select name="resolution">
<option value="640,480">640x480</option>
<option value="1280,720">1280x720</option>
<option value="1920,1080">1920x1080</option>
</select>
<button type="submit">Update Settings</button>
</form>
</div>
</div>
<script>

                document.getElementById('settingsForm').addEventListener('submit', async (e) => {

                    e.preventDefault();

                    const formData = new FormData(e.target);

                    const resolution = formData.get('resolution').split(',').map(Number);

                    const response = await fetch('/settings', {

                        method: 'PUT',

                        headers: { 'Content-Type': 'application/json' },

                        body: JSON.stringify({ resolution })

                    });

                    if (response.ok) {

                        alert('Settings updated successfully!');

                    } else {

                        alert('Failed to update settings');

                    }

                });
</script>
</body>
</html>

    """
 