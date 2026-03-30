import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Sağlık46 | Giresun Üniversitesi", layout="wide")

# --- 2. HATA GİDERİCİ (CUSTOM OBJECTS) ---
class FixedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# --- 3. MODEL YÜKLEME FONKSİYONU ---
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

# --- 4. ÜST BAŞLIK VE AKADEMİK BİLGİLER ---
st.header("Sağlık46: Meme Kanseri Teşhis Sistemi")
st.write(f"Araştırmacı: Emine Berk (2207060044) | Danışman: Dr. Öğr. Üyesi Muhammet Çakmak | Giresun Üniversitesi")
st.divider()

# --- 5. TEKNİK ÖZET VE CNN KATMANLARI ---
with st.expander("Metodoloji ve CNN Mimari Özeti", expanded=False):
    col_tech1, col_tech2 = st.columns(2)
    with col_tech1:
        st.write("**Proje Hakkında**")
        st.write("Bu sistem, ultrason görüntülerini derin öğrenme algoritmalarıyla analiz ederek teşhis desteği sağlamak üzere tasarlanmıştır.")
    with col_tech2:
        st.write("**CNN Katman Yapısı**")
        st.write("- **Convolutional (Evrişim):** Görüntüden patolojik özellikleri çıkarır.")
        st.write("- **Pooling (Havuzlama):** Veriyi sadeleştirerek işlem yükünü optimize eder.")
        st.write("- **Dense & Softmax:** Özellikleri sınıflandırarak güven oranı üretir.")

# --- 6. EĞİTİM GRAFİKLERİ (YAN YANA VE DÜZELTİLMİŞ) ---
st.subheader("Model Eğitim Performansı")
# Görsellerin yan yana durmasını kesinleştirmek için geniş kolonlar
col_g1, col_g2 = st.columns(2)

# GitHub'daki GERÇEK dosya isimlerin
grafik_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' 
karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'

with col_g1:
    st.write("**Eğitim Başarı ve Kayıp Grafiği**")
    if os.path.exists(grafik_yolu):
        # Accuracy grafiği buraya gelecek
        st.image(grafik_yolu, use_container_width=True)
    else:
        st.error("Grafik dosyası bulunamadı.")

with col_g2:
    st.write("**Karmaşıklık Matrisi (Confusion Matrix)**")
    if os.path.exists(karma_yolu):
        # Confusion Matrix buraya gelecek
        st.image(karma_yolu, use_container_width=True)
    else:
        st.error("Matris dosyası bulunamadı.")

st.divider()

# --- 7. ANALİZ ALANI ---
st.subheader("Görüntü Analizi")
col_input, col_output = st.columns([1, 1])

with col_input:
    st.write("**Görüntü Yükleme**")
    file = st.file_uploader("Ultrason görüntüsü yükleyiniz", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Yüklenen Görüntü', use_container_width=True)

with col_output:
    st.write("**Analiz Sonucu**")
    if file and model:
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Teşhisi Başlat"):
            with st.spinner('Yapay zeka analiz ediyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                guven = np.max(preds) * 100
                
                st.metric(label="Sistem Tahmini", value=classes[res_idx])
                st.write(f"**Güven Oranı:** %{guven:.2f}")
                st.progress(int(guven))
                
                if res_idx == 1:
                    st.error("Kritik Bulgular: Uzman hekim incelemesi önerilir.")
                else:
                    st.success("Düşük Risk: Bulgular stabil görünmektedir.")

# --- 8. KAYNAKÇA (GÜNCELLENDİ) ---
st.divider()
with st.expander("Akademik Kaynakça"):
    st.caption("1. Al-Dhabyani, W., et al. (2020). Dataset of breast ultrasound images. Data in Brief.")
    st.caption("2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.")
    st.caption("3. He, K., et al. (2016). Deep Residual Learning for Image Recognition.")
    st.caption("4. Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature.")

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
