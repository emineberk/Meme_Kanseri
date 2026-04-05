import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="BreastScan AI | Giresun Üniversitesi", layout="wide")

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

# --- 4. ÜST BAŞLIK (YENİ İSİM) ---
st.title("🔬 BreastScan AI: Meme Kanseri Teşhis Sistemi")
st.write(f"**Araştırmacı:** Emine Berk (2207060044) | **Danışman:** Dr. Öğr. Üyesi Muhammet Çakmak")
st.divider()

# --- 5. METODOLOJİ VE PERFORMANS ---
with st.expander("Sistem Mimarisi ve Eğitim Başarımı", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### CNN Mimari Özellikleri")
        st.write("""
        * **Özellik Çıkarımı:** Çok katmanlı Evrişimli Sinir Ağları (CNN).
        * **Optimizasyon:** Adam Optimizer & Binary Crossentropy.
        * **Veri Seti:** 128x128 boyutunda normalize edilmiş ultrason görüntüleri.
        """)
    with col_b:
        st.write("### Model Metrikleri")
        st.success("**Eğitim Accuracy:** %92.4")
        st.info("**Doğrulama Accuracy:** %77.1")

st.divider()

# --- 6. PERFORMANS GRAFİKLERİ (GÖRSELDEKİ SIRALAMA) ---
st.subheader("Model Performans Analizi")
col_g1, col_g2, col_g3 = st.columns(3)

# Dosya Yolları
kayip_yolu = 'Ekran görüntüsü 2026-04-05 172317.png'
dogruluk_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'
karma_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' 

with col_g1:
    st.write("**Eğitim/Doğrulama Kaybı (Loss)**")
    if os.path.exists(kayip_yolu):
        st.image(kayip_yolu, use_container_width=True)
    else: st.error("Dosya bulunamadı.")

with col_g2:
    st.write("**Eğitim/Doğrulama Doğruluğu (Accuracy)**")
    if os.path.exists(dogruluk_yolu):
        st.image(dogruluk_yolu, use_container_width=True)
    else: st.error("Dosya bulunamadı.")

with col_g3:
    st.write("**Karmaşıklık Matrisi (Confusion Matrix)**")
    if os.path.exists(karma_yolu):
        st.image(karma_yolu, use_container_width=True)
    else: st.error("Dosya bulunamadı.")

st.divider()

# --- 7. ANALİZ VE TAHMİN MEKANİZMASI ---
st.subheader("Canlı Görüntü Analizi")
c1, c2 = st.columns([1, 1])

with c1:
    st.write("**Ultrason Görüntüsü Yükleme**")
    file = st.file_uploader("Dosya seçiniz (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Analiz Edilecek Görüntü', use_container_width=True)

with c2:
    st.write("**Yapay Zeka Karar Çıktısı**")
    if file and model:
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Analizi Başlat"):
            with st.spinner('Model hesaplama yapıyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                guven = np.max(preds) * 100
                
                # Güvenlik Filtresi (%75)
                if guven < 75.0:
                    st.error("⚠️ **Düşük Güven Skoru!**")
                    st.warning("Yüklenen görsel tıbbi bir doku yapısı olarak tanımlanamadı. Lütfen geçerli bir ultrason yükleyin.")
                else:
                    st.metric("Tahmin Edilen Sınıf", classes[res_idx])
                    st.write(f"**Güven Oranı:** %{guven:.2f}")
                    st.progress(int(guven))
                    
                    if res_idx == 1:
                        st.error("🚨 KRİTİK: Malignant (Kötü Huylu) bulgu saptandı.")
                    else:
                        st.success("✅ STABİL: Belirgin bir risk saptanmadı.")

# --- 8. FOOTER ---
st.divider()
st.caption("© 2026 BreastScan AI | Giresun Üniversitesi | Bilgisayar Mühendisliği")
