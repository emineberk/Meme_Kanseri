import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Sağlık46 | Giresun Üni.", page_icon="🏥", layout="wide")

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
        return tf.keras.models.load_model(model_yolu, custom_objects={'Dense': FixedDense}, compile=False)
    except: return None

model = model_getir()

# --- 4. ÜST BAŞLIK VE ÖZET ---
st.title("🩺 Sağlık46: Meme Kanseri Teşhis Sistemi")
st.write(f"**Araştırmacı:** Emine Berk (2207060044) | **Danışman:** Dr. Öğr. Üyesi Muhammet Çakmak")
st.markdown("---")

with st.expander("📖 Proje Özeti ve Metodoloji (Tıklayın)", expanded=False):
    st.write("""
    **Proje Amacı:** Bu çalışma, derin öğrenme algoritmaları kullanarak ultrason görüntülerinden meme kanseri teşhisine yardımcı olmayı amaçlamaktadır. 
    **Yöntem:** Giresun Üniversitesi bünyesinde yürütülen bu çalışmada, evrişimli sinir ağları (CNN) mimarisi kullanılmıştır. 
    **Veri Seti:** Model; Normal, İyi Huylu (Benign) ve Kötü Huylu (Malignant) olmak üzere 3 sınıfta eğitilmiştir.
    """)

# --- 5. EĞİTİM GRAFİKLERİ (SUNUM İÇİN KRİTİK) ---
with st.expander("📊 Model Eğitim Performansı (Görseller)", expanded=False):
    st.write("**Açıklama:** Modelin eğitim sürecindeki başarı (Accuracy) ve kayıp (Loss) grafikleri aşağıdadır.")
    
    # GitHub'daki dosya adlarıyla birebir aynı olmalı
    grafik_yolu = 'Meme_Kanseri_Egitim_Grafigi.png'
    karma_yolu = 'Karma_Matrisi.png' # Veya ConfusuionMatrix, Conf_Matrix vb.
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(grafik_yolu):
            img_acc = Image.open(grafik_yolu)
            st.image(img_acc, caption='Eğitim Başarı ve Kayıp Grafiği', use_container_width=True)
        else:
            st.warning(f"{grafik_yolu} bulunamadı.")
            
    with col2:
        if os.path.exists(karma_yolu):
            img_cm = Image.open(karma_yolu)
            st.image(img_cm, caption='Karmaşıklık Matrisi (Confusion Matrix)', use_container_width=True)
        else:
            st.warning(f"{karma_yolu} bulunamadı.")

st.markdown("---")

# --- 6. ANALİZ ALANI ---
st.subheader("Görüntü Analizi ve Teşhis")
col1, col2 = st.columns([1, 1])

with col1:
    st.write("**📤 Görüntü Yükleme**")
    file = st.file_uploader("Ultrason Görüntüsü Seçin", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Yüklenen Görüntü', use_container_width=True)

with col2:
    st.write("**🔬 Analiz Sonucu**")
    if file and model:
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("🔎 Teşhis Koy"):
            preds = model.predict(img_array)
            classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
            res_idx = np.argmax(preds)
            oran = np.max(preds) * 100
            
            st.metric("Tahmin", classes[res_idx])
            st.write(f"**Güven Oranı:** %{oran:.2f}")
            st.progress(int(oran))
            
            if res_idx == 1: st.error("🚨 Yüksek Risk: Klinik inceleme gereklidir.")
            else: st.success("✅ Düşük Risk: Normal bulgular.")

# --- 7. KAYNAKÇA ---
st.markdown("---")
with st.expander("📚 Kaynakça"):
    st.write("""
    1. Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images. Data in Brief.
    2. Chollet, F. (2017). Deep Learning with Python. Manning Publications.
    3. Streamlit Documentation: https://docs.streamlit.io
    4. Giresun Üniversitesi Bitirme Projesi Kılavuzu.
    """)

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
