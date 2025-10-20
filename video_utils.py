import cv2
import os
from moviepy.editor import ImageSequenceClip
from PIL import Image
import numpy as np
import tempfile

def process_video(model, input_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output, _ = model.enhance(frame_rgb)
        frames.append(Image.fromarray(output))

    cap.release()

    # Geçici dosya oluştur
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        clip = ImageSequenceClip([np.array(f) for f in frames], fps=fps)
        clip.write_videofile(tmp.name, codec="libx264", audio=False)
        return tmp.name
