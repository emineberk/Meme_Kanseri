import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI VE AKADEMIK TEMA ---
st.set_page_config(page_title="Sağlık46 | Giresun Üniversitesi", layout="wide")

# Daha profesyonel bir görünüm için özel CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-family: 'Times New Roman', serif;
        color: #FFFFFF;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        font-family: 'Arial', sans-serif;
        color: #BBBBBB;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .report-card {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #333;
        border-top: 4px solid #4A90E2;
    }
    .stExpander {
        border: 1px solid #333;
        border-radius: 8px;
    }
    </style>
""", unsafe_mode=True)

# --- 2. HATA GIDERICI KATMAN ---
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
        return tf.keras.models.load_model(
            model_yolu, 
            custom_objects={'Dense': FixedDense}, 
            compile=False, 
            safe_mode=False
        )
    except: return None

model = model_getir()

# --- 4. AKADEMIK BAŞLIK ---
st.markdown("<h1 class='main-title'>Sağlık46: Meme Kanseri Teşhis Sistemi</h1>", unsafe_mode=True)
st.markdown("<p class='sub-title'>Giresun Üniversitesi Mühendislik Fakültesi<br>Araştırmacı: Emine Berk (2207060044) | Danışman: Dr. Öğr. Üyesi Muhammet Çakmak</p>", unsafe_mode=True)

# --- 5. ÖZET VE PERFORMANS GRAFIKLERI ---
col_left, col_right = st.columns([1, 2])

with col_left:
    with st.expander("Proje Özeti ve Metodoloji", expanded=False):
        st.write("""
        **Amaç:** Derin öğrenme algoritmalarıyla ultrason görüntülerinden otomatik teşhis desteği sağlamak.
        **Yöntem:** Evrişimli Sinir Ağları (CNN) mimarisi kullanılmıştır.
        **Sınıflandırma:** Normal, İyi Huylu (Benign) ve Kötü Huylu (Malignant) sınıfları tanımlanmıştır.
        """)

with col_right:
    with st.expander("Model Eğitim Performansı", expanded=True):
        # GitHub'daki gerçek dosya isimlerin
        grafik_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' 
        karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'
        
        c1, c2 = st.columns(2)
        if os.path.exists(grafik_yolu):
            c1.image(grafik_yolu, caption='Başarı ve Kayıp Grafiği', use_container_width=True)
        else:
            c1.error("Grafik dosyası bulunamadı.")
            
        if os.path.exists(karma_yolu):
            c2.image(karma_yolu, caption='Karmaşıklık Matrisi', use_container_width=True)
        else:
            c2.error("Matris dosyası bulunamadı.")

st.divider()

# --- 6. ANALIZ ALANI ---
st.subheader("Görüntü Analizi ve Teşhis")
col_input, col_output = st.columns([1, 1])

with col_input:
    st.write("**Görüntü Yükleme**")
    file = st.file_uploader("Ultrason görüntüsünü buraya sürükleyin", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Yüklenen Ultrason Görüntüsü', use_container_width=True)

with col_output:
    st.write("**Analiz Sonucu**")
    if file and model:
        st.markdown("<div class='report-card'>", unsafe_mode=True)
        
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Analizi Gerçekleştir"):
            with st.spinner('İşleniyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                oran = np.max(preds) * 100
                
                st.metric("Tahmin Edilen Sınıf", classes[res_idx])
                st.write(f"**Güven Oranı:** %{oran:.2f}")
                st.progress(int(oran))
                
                if res_idx == 1:
                    st.error("Kritik Bulgu: Klinik inceleme ve uzman görüşü gereklidir.")
                else:
                    st.success("Düşük Risk: Bulgular stabil görünmektedir.")
        
        st.markdown("</div>", unsafe_mode=True)

# --- 7. KAYNAKÇA ---
st.divider()
with st.expander("Kaynakça"):
    st.caption("1. Al-Dhabyani, W., et al. (2020). Dataset of breast ultrasound images.")
    st.caption("2. Chollet, F. (2017). Deep Learning with Python.")
    st.caption("3. Giresun Üniversitesi Bitirme Projesi Kılavuzu.")

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
