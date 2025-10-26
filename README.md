# 🧠 Real-ESRGAN Image & Video Upscaler

> Enhance low-resolution images and videos using **Real-ESRGAN** — a powerful generative model for super-resolution.

## ✨ Özellikler

- 🖼️ **Görüntü iyileştirme:** 4×'e kadar upscaling
- 🎞️ **Video işleme:** Frame-by-frame enhancement
- ⚙️ **Esnek altyapı:** CPU & GPU desteği
- 🌐 **Modern UI:** Gradio tabanlı kullanıcı arayüzü
- 🎯 **Yüksek kalite:** Gerçekçi detay iyileştirme
- ⚡ **Hızlı işlem:** Optimize edilmiş model mimarisi

> 💡 Bu proje Real-ESRGAN modelini kullanarak düşük çözünürlüklü görüntü ve videoları iyileştirir.

---

## 📦 İçindekiler

- [Kurulum](#-kurulum)
  - [Projeyi İndirin](#1️⃣-projeyi-i̇ndirin)
  - [Bağımlılıkları Yükleyin](#2️⃣-bağımlılıkları-yükleyin)
  - [Model Dosyasını İndirin](#3️⃣-model-dosyasını-i̇ndirin)
- [Kullanım](#️-kullanım)
  - [Yerel Çalıştırma](#yerel-çalıştırma)
- [Model Detayları](#-model-detayları)
- [Sorun Giderme](#️-sorun-giderme)
- [Lisans](#-lisans)

---

## ⚙️ Kurulum

### 1️⃣ Projeyi İndirin

```bash
git clone https://github.com/Fetiiii/real-esrgan-upscaler.git
cd real-esrgan-upscaler
```

### 2️⃣ Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

**Gerekli kütüphaneler:**
- PyTorch (2.0+)
- Gradio
- OpenCV
- NumPy
- Pillow

### 3️⃣ Model Dosyasını İndirin

Eğitilmiş model dosyasını (63 MB) indirin ve proje ana dizinine yerleştirin:

🔗 **[RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)**

Dosya yapısı:

```
real-esrgan-upscaler/
├── RealESRGAN_x4plus.pth  ← Buraya
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Kullanım

### Yerel Çalıştırma

Gradio arayüzünü başlatın:

```bash
python app.py
```

Uygulama otomatik olarak tarayıcınızda açılacaktır: **http://127.0.0.1:7860**

### Gradio Arayüzü Kullanımı

1. **Görüntü yükleyin** veya sürükleyip bırakın
2. **Upscale** butonuna tıklayın
3. **İyileştirilmiş görüntüyü** indirin

---

## 🧠 Model Detayları

| Özellik | Değer |
|---------|-------|
| **Mimari** | RRDBNet (Residual-in-Residual Dense Blocks) |
| **Ölçek Faktörü** | 4× |
| **Eğitim Veri Seti** | DF2K + sentetik degradasyonlar |
| **Framework** | PyTorch |
| **Model Boyutu** | 63 MB |
| **Giriş Formatları** | JPG, PNG, BMP, TIFF |
| **Çıkış Formatı** | PNG (kayıpsız) |

### Mimari Özellikleri

- **RRDB Blokları:** Derin residual yapı ile detay koruma
- **GAN Tabanlı:** Gerçekçi doku üretimi
- **Perceptual Loss:** İnsan algısına optimize edilmiş

---


## 🛠️ Sorun Giderme

| Sorun | Açıklama / Çözüm |
|-------|------------------|
| `Model file not found` | `RealESRGAN_x4plus.pth` dosyasının proje dizininde olduğundan emin olun. |
| `CUDA out of memory` | Daha küçük batch size kullanın veya CPU moduna geçin. |
| `Module not found` | `pip install -r requirements.txt` komutunu tekrar çalıştırın. |
| Gradio açılmıyor | Port 7860'ın kullanımda olmadığını kontrol edin veya `app.py`'de farklı port belirleyin. |
| Düşük görüntü kalitesi | Giriş görüntüsünün çok bozuk olmadığından emin olun. Model 4× ölçekleme için optimize edilmiştir. |
| `API key hatası` | Model dosyasının doğru indirildiğinden ve yolunun doğru olduğundan emin olun. |

### GPU Kullanımı

CUDA destekli GPU kullanmak için PyTorch'un GPU sürümünü yüklediğinizden emin olun:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Bellek Optimizasyonu

Büyük görüntüler için tile-based işleme kullanın:

```python
# app.py içinde tile_size ayarı
tile_size = 512  # Bellek durumuna göre ayarlayın
```

---

### Geliştirme Fikirleri

- Batch processing desteği
- Farklı upscaling ölçekleri (2×, 3×)
- Web API (FastAPI/Flask)
- Docker containerization
- Özel model eğitimi scripti

---

## 📘 Lisans

Bu proje **MIT Lisansı** altında dağıtılmaktadır.

```
MIT License © 2025 Feti
```

Pretrained model ağırlıkları [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) reposuna aittir ve orijinal lisans şartlarına tabidir.

---

## 📚 Referanslar

- **Original Paper:** [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- **GitHub Repository:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Model Documentation:** Pretrained weights documentation

---

## 🧠 Yazar Notu

**Real-ESRGAN Upscaler**, düşük çözünürlüklü görüntü ve videoları yapay zeka ile iyileştirmek için geliştirilmiş açık kaynaklı bir araçtır.

Deep learning ve computer vision altyapısı örneği olarak tasarlanmıştır.

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!**

</div>
