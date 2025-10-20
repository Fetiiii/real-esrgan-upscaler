# ============================================================
# ✅ REAL-ESRGAN LOCAL UPSCALER (FINAL STABLE VERSION)
# ============================================================

import types
import torchvision
import gradio as gr
import numpy as np
from PIL import Image
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from video_utils import process_video

# ────────────────────────────────────────────────
# 🔧 TorchVision 0.15+ compatibility patch
if not hasattr(torchvision.transforms, "functional_tensor"):
    from torchvision.transforms import functional as F
    torchvision.transforms.functional_tensor = types.SimpleNamespace(
        rgb_to_grayscale=F.rgb_to_grayscale
    )
# ────────────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🔹 Model mimarisi (Real-ESRGAN x4)
model_net = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64, num_block=23,
    num_grow_ch=32, scale=4
)

# 🔹 Model yükleme
model = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',  # Xintao releases'ten indirdiğin dosya
    dni_weight=None,
    model=model_net,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)
print("✅ Model başarıyla yüklendi ve hazır.")


# ============================================================
# 🔹 Görsel Upscale Fonksiyonu
# ============================================================
def upscale_image(input_image):
    if input_image is None:
        return "⚠️ Lütfen bir görsel yükleyin.", None
    np_img = np.array(input_image)
    output, _ = model.enhance(np_img)
    return "✅ Görsel başarıyla yükseltildi.", Image.fromarray(output)


# ============================================================
# 🔹 Video Upscale Fonksiyonu
# ============================================================
def upscale_video(input_video):
    if input_video is None:
        return "⚠️ Lütfen bir video yükleyin.", None
    try:
        output_path = process_video(model, input_video)
        return "✅ Video başarıyla yükseltildi.", output_path
    except Exception as e:
        return f"❌ Hata: {str(e)}", None


# ============================================================
# 🔹 Gradio Arayüzü
# ============================================================
with gr.Blocks(title="AI Image & Video Upscaler (Real-ESRGAN)") as demo:
    gr.Markdown(
        "# 🧠 AI Image & Video Upscaler\n"
        "Yapay zekâ ile fotoğraf ve videoların çözünürlüğünü artırın."
    )

    with gr.Tab("📸 Görsel"):
        img_input = gr.Image(type="pil", label="Görsel Yükle")
        img_output_text = gr.Textbox(label="Durum")
        img_output_image = gr.Image(label="Yükseltilmiş Görsel")
        img_button = gr.Button("Görseli Yükselt")
        img_button.click(
            upscale_image,
            inputs=img_input,
            outputs=[img_output_text, img_output_image],
        )

    with gr.Tab("🎞️ Video"):
        vid_input = gr.Video(label="Video Yükle (MP4)")
        vid_output_text = gr.Textbox(label="Durum")
        vid_output_video = gr.Video(label="Yükseltilmiş Video")
        vid_button = gr.Button("Videoyu Yükselt")
        vid_button.click(
            upscale_video,
            inputs=vid_input,
            outputs=[vid_output_text, vid_output_video],
        )

demo.launch()
