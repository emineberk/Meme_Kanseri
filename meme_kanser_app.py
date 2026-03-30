import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Sağlık46 | Giresun Üniversitesi", layout="wide")

# GRAFİKLERİ YAN YANA ZORLAYAN ÖZEL CSS
st.markdown("""
    <style>
    /* Sütunların yan yana durmasını ve birbirinin altına kaymamasını sağlar */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: flex-start !important;
    }
    [data-testid="stColumn"] {
        min-width: 45% !important;
    }
    .main-title { text-align: center; font-family: serif; color: #FFFFFF; margin-bottom: 0px; }
    .sub-title { text-align: center; font-family: sans-serif; color: #BBBBBB; font-size: 1.1rem; margin-bottom: 20px; }
    </style>
""", unsafe_mode=True)

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

# --- 4. AKADEMİK ÜST BAŞLIK ---
st.markdown("<h1 class='main-title'>Sağlık46: Meme Kanseri Teşhis Sistemi</h1>", unsafe_mode=True)
st.markdown("<p class='sub-title'>Giresun Üniversitesi Mühendislik Fakültesi<br>Araştırmacı: Emine Berk (2207060044) | Danışman: Dr. Öğr. Üyesi Muhammet Çakmak</p>", unsafe_mode=True)
st.divider()

# --- 5. DETAYLI ÖZET VE METODOLOJİ ---
with st.expander("Proje Detayları ve CNN Metodolojisi", expanded=False):
    st.write("### Proje Hakkında")
    st.write("Bu sistem, ultrason görüntülerini derin öğrenme algoritmalarıyla analiz ederek teşhis desteği sağlamak üzere tasarlanmıştır.")
    c_met1, c_met2 = st.columns(2)
    with c_met1:
        st.write("**Teknik Katman Analizi**")
        st.info("Convolutional (Evrişim) katmanları görüntünün doku özelliklerini çıkarırken, Pooling katmanları veriyi optimize eder.")
    with c_met2:
        st.write("**Sınıflandırma Mantığı**")
        st.success("Model; Normal, Benign (İyi Huylu) ve Malignant (Kötü Huylu) olmak üzere 3 sınıfta teşhis koyar.")

# --- 6. MODEL PERFORMANSI (YAN YANA SABİTLENDİ) ---
st.subheader("Model Eğitim Performans Verileri")

# GitHub'daki dosya yolların
grafik_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' # Accuracy/Loss grafiği
karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'   # Confusion Matrix

# Bu kısım görselleri yan yana getiren ana bloktur
col_g1, col_g2 = st.columns(2, gap="medium")

with col_g1:
    st.write("**Öğrenme Eğrileri (Accuracy & Loss)**")
    if os.path.exists(grafik_yolu):
        st.image(grafik_yolu, use_container_width=True)
    else:
        st.error("Accuracy grafiği bulunamadı.")

with col_g2:
    st.write("**Hata Analizi (Confusion Matrix)**")
    if os.path.exists(karma_yolu):
        st.image(karma_yolu, use_container_width=True)
    else:
        st.error("Matris dosyası bulunamadı.")

st.divider()

# --- 7. ANALİZ ALANI ---
st.subheader("Görüntü Analizi ve Canlı Teşhis")
c1, c2 = st.columns([1, 1])

with c1:
    st.write("**Veri Girişi**")
    file = st.file_uploader("Ultrason görüntüsü yükleyiniz", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Sisteme Aktarılan Görüntü', use_container_width=True)

with c2:
    st.write("**Yapay Zeka Karar Mekanizması**")
    if file and model:
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Analizi Başlat"):
            with st.spinner('Pikseller analiz ediliyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                guven = np.max(preds) * 100
                
                st.metric("Tahmin Edilen Sınıf", classes[res_idx])
                st.write(f"**Güven Oranı:** %{guven:.2f}")
                st.progress(int(guven))
                
                if res_idx == 1:
                    st.error("Kritik Uyarı: Malignant bulgu tespit edildi. Klinik inceleme önerilir.")
                else:
                    st.success("Düşük Risk: Bulgular stabil değerlendirilmiştir.")

# --- 8. AKADEMİK KAYNAKÇA ---
st.divider()
with st.expander("Akademik Referanslar"):
    st.caption("1. Al-Dhabyani, W., et al. (2020). Dataset of breast ultrasound images. Data in Brief.")
    st.caption("2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.")
    st.caption("3. Esteva, A., et al. (2017). Dermatologist-level classification with deep neural networks. Nature.")

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
