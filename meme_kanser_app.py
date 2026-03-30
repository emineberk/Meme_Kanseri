import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI VE AKADEMİK TEMA ---
st.set_page_config(page_title="Sağlık46 | Giresun Üniversitesi", layout="wide")

# Görselliği artırmak ve grafikleri yan yana sabitlemek için özel CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-family: 'Times New Roman', serif;
        color: #FFFFFF;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        font-family: 'Arial', sans-serif;
        color: #BBBBBB;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .stExpander {
        border: 1px solid #333;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .report-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
    }
    </style>
""", unsafe_mode=True)

# --- 2. MODEL UYUMLULUK KATMANI ---
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

# --- 4. AKADEMİK ÜST BİLGİ ---
st.markdown("<h1 class='main-title'>Sağlık46: Meme Kanseri Teşhis Sistemi</h1>", unsafe_mode=True)
st.markdown("<p class='sub-title'>Giresun Üniversitesi Mühendislik Fakültesi<br>Araştırmacı: Emine Berk (2207060044) | Danışman: Dr. Öğr. Üyesi Muhammet Çakmak</p>", unsafe_mode=True)

# --- 5. ÖZET VE PERFORMANS (YAN YANA) ---
col_left, col_right = st.columns([1, 2])

with col_left:
    with st.expander("📖 Proje Özeti", expanded=False):
        st.write("""
        **Metodoloji:** Bu projede evrişimli sinir ağları (CNN) kullanılarak tıbbi görüntü işleme teknikleri uygulanmıştır. 
        **Kapsam:** Ultrason görüntüleri üzerinden Normal, Benign ve Malignant sınıfları için yüksek doğruluklu tahminleme hedeflenmiştir.
        """)

with col_right:
    with st.expander("📊 Eğitim Performansı", expanded=True):
        # GitHub'daki GERÇEK dosya isimlerin
        g1 = 'Ekran görüntüsü 2026-03-29 231910.png' 
        g2 = 'Ekran görüntüsü 2026-03-29 232001.png'
        
        c1, c2 = st.columns(2)
        if os.path.exists(g1):
            c1.image(g1, caption='Eğitim Grafikleri', use_container_width=True)
        else:
            c1.warning("Grafik 1 yüklenemedi.")
            
        if os.path.exists(g2):
            c2.image(g2, caption='Karmaşıklık Matrisi', use_container_width=True)
        else:
            c2.warning("Grafik 2 yüklenemedi.")

st.divider()

# --- 6. ANALİZ VE TEŞHİS ALANI ---
st.subheader("Görüntü Analizi")
col_upload, col_result = st.columns([1, 1])

with col_upload:
    file = st.file_uploader("Analiz edilecek ultrason görüntüsünü seçiniz", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Seçilen Görüntü', use_container_width=True)

with col_result:
    if file and model:
        st.markdown("<div class='report-card'>", unsafe_mode=True)
        st.write("### Analiz Raporu")
        
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Analizi Gerçekleştir"):
            with st.spinner('Yapay zeka katmanları işleniyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                oran = np.max(preds) * 100
                
                st.metric("Sistem Tahmini", classes[res_idx])
                st.write(f"**Doğruluk Olasılığı:** %{oran:.2f}")
                st.progress(int(oran))
                
                if res_idx == 1:
                    st.error("Kritik Bulgular: Uzman hekim onayı gereklidir.")
                else:
                    st.success("Normal/Düşük Riskli bulgular tespit edildi.")
        st.markdown("</div>", unsafe_mode=True)

# --- 7. KAYNAKÇA ---
st.divider()
with st.expander("📚 Kaynakça"):
    st.caption("1. Al-Dhabyani, W., et al. (2020). Dataset of breast ultrasound images. Data in Brief.")
    st.caption("2. Chollet, F. (2017). Deep Learning with Python. Manning Publications.")
    st.caption("3. Giresun Üniversitesi Bitirme Projesi Yazım Kılavuzu.")

st.caption("© 2026 Sağlık46 Projesi | Giresun Üniversitesi")
