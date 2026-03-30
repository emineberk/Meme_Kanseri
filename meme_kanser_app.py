import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI VE AKADEMİK TEMA ---
st.set_page_config(page_title="Sağlık46 | Giresun Üniversitesi", layout="wide")

# Grafikleri yan yana sabitlemek ve modern görünüm için CSS
st.markdown("""
    <style>
    .main-title { text-align: center; font-family: serif; color: #FFFFFF; }
    .sub-title { text-align: center; font-family: sans-serif; color: #BBBBBB; font-size: 1.1rem; margin-bottom: 20px; }
    .report-card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-top: 4px solid #4A90E2; }
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

# --- 4. ÜST BAŞLIK ---
st.markdown("<h1 class='main-title'>Sağlık46: Meme Kanseri Teşhis Sistemi</h1>", unsafe_mode=True)
st.markdown("<p class='sub-title'>Giresun Üniversitesi Mühendislik Fakültesi<br>Araştırmacı: Emine Berk (2207060044) | Danışman: Dr. Öğr. Üyesi Muhammet Çakmak</p>", unsafe_mode=True)
st.divider()

# --- 5. ÖZET VE CNN KATMANLARI ---
with st.expander("Proje Özeti ve Teknik Metodoloji (CNN Mimari Özeti)", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("""
        **Genel Amaç:** Ultrason görüntüleri üzerinden derin öğrenme ile kanser teşhis desteği sağlamak.
        **Veri Seti:** Normal, Benign (İyi Huylu) ve Malignant (Kötü Huylu) sınıfları kullanılmıştır.
        """)
    with col_b:
        st.write("""
        **CNN Mimari Katmanları:**
        * **Convolutional (Evrişim) Katmanı:** Görüntüden kenar ve doku özelliklerini çıkarır.
        * **Pooling (Havuzlama) Katmanı:** Veriyi sadeleştirir, işlem yükünü azaltır.
        * **Flatten & Dense:** Özellikleri tek boyuta indirip nihai sınıflandırmayı yapar.
        * **Softmax:** Tahmin edilen sınıfın olasılık değerini (Güven Oranı) üretir.
        """)

# --- 6. EĞİTİM GRAFİKLERİ (YAN YANA) ---
with st.expander("Model Eğitim Performansı (Grafikler)", expanded=True):
    # GitHub'daki GERÇEK dosya isimlerin
    grafik_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' 
    karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'
    
    col_g1, col_g2 = st.columns(2) # GRAFİKLERİ YAN YANA GETİREN KOMUT
    
    with col_g1:
        if os.path.exists(grafik_yolu):
            st.image(grafik_yolu, caption='Başarı ve Kayıp Grafiği', use_container_width=True)
        else:
            st.error("Grafik 1 bulunamadı.")
            
    with col_g2:
        if os.path.exists(karma_yolu):
            st.image(karma_yolu, caption='Karmaşıklık Matrisi (Confusion Matrix)', use_container_width=True)
        else:
            st.error("Grafik 2 bulunamadı.")

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
        
        if st.button("Teşhis Koy"):
            with st.spinner('Analiz ediliyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                oran = np.max(preds) * 100
                
                st.metric("Sistem Tahmini", classes[res_idx])
                st.write(f"**Doğruluk Olasılığı:** %{oran:.2f}")
                st.progress(int(oran))
                
                if res_idx == 1:
                    st.error("Kritik Bulgu: Klinik inceleme ve uzman radyolog onayı gereklidir.")
                else:
                    st.success("Normal/Düşük Riskli bulgular tespit edildi.")
        st.markdown("</div>", unsafe_mode=True)

# --- 8. KAYNAKÇA ---
st.divider()
with st.expander("Kaynakça"):
    st.caption("1. Al-Dhabyani, W., et al. (2020). Dataset of breast ultrasound images.")
    st.caption("2. Chollet, F. (2017). Deep Learning with Python.")
    st.caption("3. Giresun Üniversitesi Bitirme Projesi Kılavuzu.")

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
