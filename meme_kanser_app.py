import streamlit as st
import os
import numpy as np
from PIL import Image, ImageStat
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Sağlık46 | Giresun Üniversitesi", layout="wide")

# --- 2. HATA GİDERİCİ (CUSTOM OBJECTS) ---
class FixedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# --- 3. MODEL YÜKLEME ---
@st.cache_resource
def model_getir():
    model_yolu = 'Meme_Kanseri_Final_Modeli.h5'
    if not os.path.exists(model_yolu):
        file_id = '14OW6zCuzug3Ge7dZoqPuKrbFuOXr6meb'
        drive_url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(drive_url, model_yolu, quiet=False)
        except: return None
    try:
        return tf.keras.models.load_model(model_yolu, custom_objects={'Dense': FixedDense}, compile=False, safe_mode=False)
    except: return None

model = model_getir()

# --- 4. ÜST BAŞLIK ---
st.title("Sağlık46: Meme Kanseri Teşhis Sistemi")
st.write(f"**Araştırmacı:** Emine Berk (2207060044) | **Danışman:** Dr. Öğr. Üyesi Muhammet Çakmak")
st.divider()

# --- 5. METODOLOJİ ---
with st.expander("Metodoloji ve Başarı Oranları", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### CNN Katman Mimarisi")
        st.write("Evrişim (Conv2D), Havuzlama (MaxPooling) ve Dropout katmanları ile optimize edilmiştir.")
    with col_b:
        st.write("### Başarı İstatistikleri")
        st.success("**Eğitim:** %92.4 | **Doğrulama:** %77.1")

st.divider()

# --- 6. PERFORMANS GRAFİKLERİ ---
st.subheader("Model Eğitim Performans Grafikleri")
col_g1, col_g2, col_g3 = st.columns(3)

kayip_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' 
dogruluk_yolu = 'Ekran görüntüsü 2026-04-05 172317.png' 
karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'

for col, path, title in zip([col_g1, col_g2, col_g3], 
                            [kayip_yolu, dogruluk_yolu, karma_yolu], 
                            ["Kayıp (Loss)", "Doğruluk (Accuracy)", "Karmaşıklık Matrisi"]):
    with col:
        st.write(f"**{title}**")
        if os.path.exists(path): st.image(path, use_container_width=True)
        else: st.error("Dosya eksik.")

st.divider()

# --- 7. ANALİZ VE REDDETME MEKANİZMASI ---
st.subheader("Görüntü Analizi ve Canlı Teşhis")
c1, c2 = st.columns([1, 1])

with c1:
    st.write("**Görüntü Yükleme**")
    file = st.file_uploader("Ultrason görüntüsü seçin", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Yüklenen Görüntü', use_container_width=True)

with c2:
    st.write("**Yapay Zeka Karar Mekanizması**")
    if file and model:
        # 1. AŞAMA: RENK ANALİZİ (Ultrasonlar genelde düşük doygunluktadır)
        stat = ImageStat.Stat(img)
        # Eğer görsel çok renkliyse (standart sapma yüksekse) ultrason değildir
        is_colored = sum(stat.stddev) > 120 

        if st.button("Teşhisi Başlat"):
            if is_colored:
                st.error("❌ **HATA: Geçersiz Görsel Tipi!**")
                st.warning("Yüklediğiniz görsel çok fazla renk içeriyor. Meme ultrasonu görüntüleri gri tonlamalı (Siyah-Beyaz) olmalıdır. Lütfen geçerli bir tıbbi görüntü yükleyin.")
            else:
                with st.spinner('Analiz ediliyor...'):
                    img_resized = img.resize((128, 128))
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    preds = model.predict(img_array)
                    classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                    res_idx = np.argmax(preds)
                    guven = np.max(preds) * 100
                    
                    # 2. AŞAMA: GÜVEN EŞİĞİ (Threshold %80)
                    if guven < 80.0:
                        st.error("⚠️ **Analiz Reddedildi!**")
                        st.info("Bu görsel bir meme ultrasonu yapısına sahip değil veya görüntü kalitesi çok düşük. Güvenli bir teşhis yapılamıyor.")
                    else:
                        st.metric("Sistem Tahmini", classes[res_idx])
                        st.write(f"**Güven Oranı:** %{guven:.2f}")
                        st.progress(int(guven))
                        
                        if res_idx == 1:
                            st.error("Kritik Uyarı: Malignant yapı tespit edildi.")
                        else:
                            st.success("Düşük Risk: Bulgular stabil.")

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
