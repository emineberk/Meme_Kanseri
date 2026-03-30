import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Sağlık46 | Emine Berk", page_icon="🏥", layout="wide")

# --- 2. BAŞLIK VE AKADEMİK BİLGİLER ---
st.title("🩺 Sağlık46: Meme Kanseri Teşhis Sistemi")
st.info("Giresun Üniversitesi Mühendislik Fakültesi | **Araştırmacı:** Emine Berk (2207060044) | **Danışman:** Dr. Öğr. Üyesi Muhammet Çakmak")

# --- 3. HATA GİDERİCİ (CUSTOM OBJECTS) ---
# Keras 3'teki 'quantization_config' hatasını bypass etmek için özel sınıf tanımı
class FixedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None) # Hataya sebep olan parametreyi siliyoruz
        super().__init__(*args, **kwargs)

# --- 4. MODELİ DRIVE'DAN İNDİRME VE YÜKLEME ---
@st.cache_resource
def model_getir():
    model_yolu = 'Meme_Kanseri_Final_Modeli.h5'
    
    if not os.path.exists(model_yolu):
        with st.spinner('Model dosyası buluttan çekiliyor, lütfen bekleyiniz...'):
            file_id = '14OW6zCuzug3Ge7dZoqPuKrbFuOXr6meb'
            drive_url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(drive_url, model_yolu, quiet=False)
            except Exception as e:
                st.error(f"Model indirilirken hata oluştu: {e}")
                return None

    try:
        # Modeli yüklerken 'Dense' katmanını bizim düzelttiğimiz 'FixedDense' ile değiştiriyoruz
        custom_objects = {'Dense': FixedDense}
        model = tf.keras.models.load_model(
            model_yolu, 
            custom_objects=custom_objects, 
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Sistemsel bir hata oluştu: {e}")
        return None

# Modeli çağır
model = model_getir()

# --- 5. YAN PANEL VE ANALİZ ALANI ---
st.sidebar.header("Sistem Durumu")
if model:
    st.sidebar.success("✅ Sistem Analize Hazır!")
else:
    st.sidebar.error("❌ Model Yüklenemedi")

st.subheader("Görüntü Analizi ve Teşhis")
file = st.file_uploader("Analiz için Ultrason Görüntüsü Seçin", type=["jpg", "png", "jpeg"])

if file:
    col1, col2 = st.columns([1, 1])
    img = Image.open(file).convert('RGB')
    
    with col1:
        st.image(img, caption='Yüklenen Görüntü', use_container_width=True)
    
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with col2:
        if st.button("🔎 Analizi Başlat"):
            if model is not None:
                with st.spinner('Yapay zeka analiz ediyor...'):
                    preds = model.predict(img_array)
                    classes = ['Benign (İyi Huylu)', 'Malignant (Kötü Huylu)', 'Normal']
                    res_idx = np.argmax(preds)
                    st.markdown("### Sonuç Raporu")
                    st.subheader(f"Tahmin: **{classes[res_idx]}**")
                    st.write(f"Güven Oranı: %{np.max(preds)*100:.2f}")
                    st.progress(int(np.max(preds)*100))
                    
                    if res_idx == 1: st.error("🚨 Şüpheli bulgu! Doktora danışın.")
                    else: st.success("✅ Bulgular düşük risk grubunda.")
            else:
                st.error("Model yüklenemediği için analiz yapılamıyor.")
