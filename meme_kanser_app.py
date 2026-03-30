import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI VE AKADEMİK TEMA ---
st.set_page_config(page_title="Sağlık46 | Giresun Üniversitesi", layout="wide")

# Görsellerin yan yana gelmesini kesinleştirmek için özel CSS
st.markdown("""
    <style>
    .main-title { text-align: center; font-family: serif; color: #FFFFFF; }
    .sub-title { text-align: center; font-family: sans-serif; color: #BBBBBB; font-size: 1.1rem; margin-bottom: 20px; }
    .report-card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-top: 4px solid #4A90E2; }
    
    /* Sunumda grafiklerin yan yana durmasını kesinleştirir */
    [data-testid="stColumn"] {
        min-width: 450px;
    }
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
        # safe_mode=False ekleyerek tüm yükleme hatalarını bypass ediyoruz
        return tf.keras.models.load_model(model_yolu, custom_objects={'Dense': FixedDense}, compile=False, safe_mode=False)
    except: return None

model = model_getir()

# --- 4. ÜST BAŞLIK ---
st.markdown("<h1 class='main-title'>Sağlık46: Meme Kanseri Teşhis Sistemi</h1>", unsafe_mode=True)
st.markdown("<p class='sub-title'>Giresun Üniversitesi Mühendislik Fakültesi<br>Araştırmacı: Emine Berk (2207060044) | Danışman: Dr. Öğr. Üyesi Muhammet Çakmak</p>", unsafe_mode=True)
st.divider()

# --- 5. TEKNİK ÖZET ---
with st.expander("Metodoloji ve Teknik Metodoloji (CNN Özeti)", expanded=False):
    st.write("""
    **Genel Yaklaşım:** Evrişimli Sinir Ağları (CNN) mimarisi kullanılarak ultrason görüntülerinden patolojik sınıflandırma yapılmıştır.
    **CNN Katmanları:** Kenar ve doku özelliklerini yakalamak için Convolutional, veriyi sadeleştirmek için Pooling ve nihai olasılık dağılımını üretmek için Softmax katmanları kullanılmıştır.
    """)

# --- 6. EĞİTİM GRAFİKLERİ (YERLERİ DÜZELTİLDİ VE YAN YANA GETİRİLDİ) ---
st.write("### Model Eğitim Performansı")
# columns(2) kullanarak yan yana gelmesini kesinleştiriyoruz
col_grafik1, col_grafik2 = st.columns(2)

# GitHub'daki GERÇEK dosya isimlerin
grafik_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' 
karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'

with col_grafik1:
    st.write("**Eğitim Başarı ve Kayıp Grafiği**")
    if os.path.exists(grafik_yolu):
        # image_3.png'de Confusion Matrix duruyordu, buraya Accuracy grafiğini getirdik.
        st.image(grafik_yolu, use_container_width=True)
    else:
        st.error("Grafik dosyası bulunamadı.")
        
with col_grafik2:
    st.write("**Karmaşıklık Matrisi (Confusion Matrix)**")
    if os.path.exists(karma_yolu):
        # image_3.png'de Accuracy grafiği duruyordu, buraya Confusion Matrix'i getirdik.
        st.image(karma_yolu, use_container_width=True)
    else:
        st.error("Matris dosyası bulunamadı.")

st.divider()

# --- 7. ANALİZ ALANI ---
st.subheader("Görüntü Analizi ve Teşhis")
col1, col2 = st.columns([1, 1])

with col1:
    st.write("**Görüntü Yükleme**")
    file = st.file_uploader("Ultrason Görüntüsü Seçin", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Yüklenen Görüntü', use_container_width=True)

with col2:
    st.write("**Analiz Sonucu**")
    if file and model:
        st.markdown("<div class='report-card'>", unsafe_mode=True)
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Teşhisi Başlat"):
            with st.spinner('Analiz ediliyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                oran = np.max(preds) * 100
                
                st.metric("Sistem Tahmini", classes[res_idx])
                st.write(f"**Güven Oranı:** %{oran:.2f}")
                st.progress(int(oran))
                
                if res_idx == 1:
                    st.error("🚨 Kritik Bulgu: İleri tetkik gereklidir.")
                else:
                    st.success("✅ Bulgular stabil değerlendirilmiştir.")
        st.markdown("</div>", unsafe_mode=True)

# --- 8. KAYNAKÇA ---
st.divider()
with st.expander("Akademik Kaynakça"):
    st.caption("1. Al-Dhabyani, W., et al. (2020). Dataset of breast ultrasound images.")
    st.caption("2. Chollet, F. (2017). Deep Learning with Python.")
    st.caption("3. Giresun Üniversitesi Bitirme Projesi Kılavuzu.")

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
