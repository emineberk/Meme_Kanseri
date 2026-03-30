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
st.title("Sağlık46: Meme Kanseri Teşhis Sistemi")
st.write(f"**Araştırmacı:** Emine Berk (2207060044) | **Danışman:** Dr. Öğr. Üyesi Muhammet Çakmak | Giresun Üniversitesi")
st.divider()

# --- 5. DETAYLI PROJE ÖZETİ VE METODOLOJİ ---
with st.expander("Proje Detayları ve CNN Metodolojisi", expanded=True):
    st.write("### Proje Hakkında")
    st.write("Bu çalışma, ultrason görüntülerindeki patolojik dokuları derin öğrenme (CNN) mimarisi ile analiz ederek sınıflandırmaktadır.")
    
    col_tech1, col_tech2 = st.columns(2)
    with col_tech1:
        st.write("**Teknik Katman Analizi**")
        st.info("Convolutional katmanları tümör sınırlarını yakalar, Pooling katmanları ise veriyi optimize ederek en belirgin özellikleri ön plana çıkarır.")
    with col_tech2:
        st.write("**Sınıflandırma Kategorileri**")
        st.success("Sistem; Normal doku, İyi Huylu (Benign) ve Kötü Huylu (Malignant) olmak üzere üç sınıfta teşhis desteği sunar.")

# --- 6. MODEL PERFORMANSI (YAN YANA GÜVENLİ DÜZEN) ---
st.write("### Model Eğitim Performans Verileri")
col_grafik1, col_grafik2 = st.columns(2)

# Dosya yolları
grafik_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' # Accuracy/Loss
karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'   # Confusion Matrix

with col_grafik1:
    st.write("**Öğrenme Eğrileri (Accuracy & Loss)**")
    if os.path.exists(grafik_yolu):
        st.image(grafik_yolu, use_container_width=True)
    else:
        st.error("Accuracy grafiği bulunamadı.")

with col_grafik2:
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
    st.write("**Görüntü Yükleme**")
    file = st.file_uploader("Analiz için ultrason görüntüsü seçin", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Sisteme Aktarılan Görüntü', use_container_width=True)

with c2:
    st.write("**Yapay Zeka Karar Mekanizması**")
    if file and model:
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Teşhisi Başlat"):
            with st.spinner('Pikseller analiz ediliyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                guven = np.max(preds) * 100
                
                st.metric("Sistem Tahmini", classes[res_idx])
                st.write(f"**Güven Oranı:** %{guven:.2f}")
                st.progress(int(guven))
                
                if res_idx == 1:
                    st.error("Kritik Uyarı: Malignant bulgu tespit edildi. İleri tetkik önerilir.")
                else:
                    st.success("Düşük Risk: Bulgular stabil değerlendirilmiştir.")

# --- 8. AKADEMİK KAYNAKÇA ---
st.divider()
with st.expander("Akademik Referanslar"):
    st.caption("1. Al-Dhabyani, W., et al. (2020). Dataset of breast ultrasound images. Data in Brief.")
    st.caption("2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.")
    st.caption("3. Esteva, A., et al. (2017). Dermatologist-level classification with deep neural networks. Nature.")

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
