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

# --- 4. ÜST BAŞLIK ---
st.title("Sağlık46: Meme Kanseri Teşhis Sistemi")
st.write(f"**Araştırmacı:** Emine Berk (2207060044) | **Danışman:** Dr. Öğr. Üyesi Muhammet Çakmak")
st.divider()

# --- 5. METODOLOJİ ---
with st.expander("Metodoloji ve Başarı Oranları", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### CNN Katman Mimarisi")
        st.write("Model; Conv2D, MaxPooling ve Dropout katmanları ile tıbbi görüntü analizi için optimize edilmiştir.")
    with col_b:
        st.write("### Başarı İstatistikleri")
        st.success("**Eğitim Başarı Oranı:** %92.4")
        st.info("**Doğrulama Başarı Oranı:** %77.1")

st.divider()

# --- 6. MODEL PERFORMANS ANALİZİ (GÖRSELDEKİ SIRALAMAYA GÖRE) ---
st.subheader("Model Eğitim Performans Grafikleri")
col_g1, col_g2, col_g3 = st.columns(3)

# Dosya Yolları
kayip_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' 
dogruluk_yolu = 'Ekran görüntüsü 2026-04-05 172317.png' 
karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'

with col_g1:
    st.write("**1. Kayıp (Loss) Grafiği**")
    if os.path.exists(kayip_yolu):
        st.image(kayip_yolu, use_container_width=True)
    else:
        st.error("Kayıp grafiği dosyası bulunamadı.")

with col_g2:
    st.write("**2. Doğruluk (Accuracy) Grafiği**")
    if os.path.exists(dogruluk_yolu):
        st.image(dogruluk_yolu, use_container_width=True)
    else:
        st.error("Doğruluk grafiği dosyası bulunamadı.")

with col_g3:
    st.write("**3. Karmaşıklık Matrisi (Confusion Matrix)**")
    if os.path.exists(karma_yolu):
        st.image(karma_yolu, use_container_width=True)
    else:
        st.error("Matris dosyası bulunamadı.")

st.divider()

# --- 7. ANALİZ ALANI VE GÜVENLİK FİLTRESİ ---
st.subheader("Görüntü Analizi ve Canlı Teşhis")
c1, c2 = st.columns([1, 1])

with c1:
    st.write("**Görüntü Yükleme**")
    file = st.file_uploader("Ultrason görüntüsü seçin", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Yüklenen Görüntü', use_container_width=True)

with c2:
    st.write("**Yapay Zeka Karar Mekanizması**")
    if file and model:
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Teşhisi Başlat"):
            with st.spinner('Analiz ediliyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                guven = np.max(preds) * 100
                
                # --- ALAKASIZ FOTOĞRAF ENGELLEME (%75 EŞİĞİ) ---
                if guven < 75.0:
                    st.error("⚠️ **Analiz Reddedildi: Geçersiz Görüntü**")
                    st.warning(f"Sistem güven oranı düşük (%{guven:.2f}). Yüklenen görsel tıbbi bir ultrason yapısı içermiyor olabilir.")
                    st.info("Lütfen net ve sadece meme ultrasonu içeren bir dosya yükleyin.")
                else:
                    st.metric("Sistem Tahmini", classes[res_idx])
                    st.write(f"**Güven Oranı:** %{guven:.2f}")
                    st.progress(int(guven))
                    
                    if res_idx == 1:
                        st.error("Kritik Uyarı: Malignant (Kötü Huylu) yapı tespit edildi. Klinik inceleme gereklidir.")
                    else:
                        st.success("Düşük Risk: Bulgular stabil değerlendirilmiştir.")

# --- 8. ALT BİLGİ ---
st.divider()
st.caption("© 2026 Sağlık46 | Giresun Üniversitesi | Mühendislik Fakültesi")
