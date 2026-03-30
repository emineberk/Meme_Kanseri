import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Sağlık46 | Giresun Üniversitesi", layout="wide")

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
        return tf.keras.models.load_model(model_yolu, custom_objects={'Dense': FixedDense}, compile=False, safe_mode=False)
    except: return None

model = model_getir()

# --- 4. AKADEMİK ÜST BİLGİ ---
st.title("Sağlık46: Meme Kanseri Teşhis Sistemi")
st.write(f"Araştırmacı: Emine Berk (2207060044) | Danışman: Dr. Öğr. Üyesi Muhammet Çakmak | Giresun Üniversitesi")
st.divider()

# --- 5. TEKNİK METODOLOJİ VE CNN ÖZETİ ---
with st.expander("Metodoloji ve CNN Mimari Özeti", expanded=False):
    col_tech1, col_tech2 = st.columns(2)
    with col_tech1:
        st.write("**Proje Amacı**")
        st.write("Bu çalışma, evrişimli sinir ağları (CNN) mimarisi kullanılarak ultrason görüntüleri üzerinden patolojik sınıflandırma yapmaktadır.")
    with col_tech2:
        st.write("**CNN Katman Yapısı**")
        st.write("- **Convolutional Layers:** Görüntüdeki mikro-kireçlenmeleri ve doku bozukluklarını tespit eder.")
        st.write("- **Pooling & Dropout:** Veri boyutunu optimize ederken aşırı öğrenmeyi (overfitting) engeller.")
        st.write("- **Dense & Softmax:** Olasılıksal bir güven oranıyla nihai teşhis sonucunu üretir.")

# --- 6. EĞİTİM GRAFİKLERİ (YAN YANA SABİTLENDİ) ---
st.write("### Model Eğitim Performansı")
# columns(2) kullanarak yan yana gelmesini kesinleştiriyoruz
col_grafik1, col_grafik2 = st.columns(2)

# GitHub'daki dosya isimlerin
g1_yolu = 'Ekran görüntüsü 2026-03-29 231910.png'
g2_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'

with col_grafik1:
    if os.path.exists(g1_yolu):
        st.image(g1_yolu, caption="Eğitim ve Doğrulama Grafiği (Accuracy/Loss)", use_container_width=True)
    else:
        st.warning("Grafik 1 bulunamadı.")

with col_grafik2:
    if os.path.exists(g2_yolu):
        st.image(g2_yolu, caption="Karmaşıklık Matrisi (Confusion Matrix)", use_container_width=True)
    else:
        st.warning("Grafik 2 bulunamadı.")

st.divider()

# --- 7. ANALİZ ALANI ---
st.subheader("Görüntü Analizi ve Teşhis")
c1, c2 = st.columns([1, 1])

with c1:
    st.write("**Görüntü Yükleme**")
    file = st.file_uploader("Ultrason Görüntüsü Seçin", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Yüklenen Görüntü', use_container_width=True)

with c2:
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
                oran = np.max(preds) * 100
                
                st.metric("Sistem Tahmini", classes[res_idx])
                st.write(f"**Güven Oranı:** %{oran:.2f}")
                st.progress(int(oran))
                
                if res_idx == 1:
                    st.error("Kritik Bulgu: Klinik inceleme ve biyopsi onayı gereklidir.")
                else:
                    st.success("Düşük Risk: Bulgular stabil değerlendirilmiştir.")

# --- 8. KAYNAKÇA ---
st.divider()
with st.expander("Akademik Kaynakça"):
    st.caption("1. Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images. Data in Brief.")
    st.caption("2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.")
    st.caption("3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.")
    st.caption("4. Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature.")

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
