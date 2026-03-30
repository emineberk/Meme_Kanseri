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
        # safe_mode=False ve custom_objects ile tüm yükleme hatalarını bypass ediyoruz
        return tf.keras.models.load_model(model_yolu, custom_objects={'Dense': FixedDense}, compile=False, safe_mode=False)
    except: return None

model = model_getir()

# --- 4. ÜST BAŞLIK VE AKADEMİK BİLGİLER ---
st.header("Sağlık46: Meme Kanseri Teşhis Sistemi")
st.caption(f"Araştırmacı: Emine Berk (2207060044) | Danışman: Dr. Öğr. Üyesi Muhammet Çakmak | Giresun Üniversitesi")
st.divider()

# --- 5. TEKNİK ÖZET VE CNN KATMANLARI ---
with st.expander("Metodoloji ve CNN Mimari Özeti", expanded=False):
    col_tech1, col_tech2 = st.columns(2)
    with col_tech1:
        st.write("**Proje Hakkında**")
        st.write("Bu sistem, ultrason görüntülerini analiz ederek üç farklı sınıfta (Normal, Benign, Malignant) teşhis desteği sağlamak üzere eğitilmiş bir Derin Öğrenme modelidir.")
    with col_tech2:
        st.write("**CNN Katman Yapısı**")
        st.write("- **Evrişim (Conv2D):** Görüntüdeki patolojik özellikleri yakalar.")
        st.write("- **Havuzlama (MaxPooling):** Önemli veriyi koruyup boyutu küçültür.")
        st.write("- **Tam Bağlantılı (Dense):** Çıkarılan özellikleri sınıflara atar.")

# --- 6. EĞİTİM GRAFİKLERİ (YAN YANA) ---
with st.expander("Model Eğitim Performansı", expanded=True):
    # GitHub'daki GERÇEK dosya isimlerin (image_967ebe.png'den alındı)
    grafik_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' 
    karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        if os.path.exists(grafik_yolu):
            st.image(grafik_yolu, caption='Accuracy ve Loss Grafikleri', use_container_width=True)
        else:
            st.warning("Eğitim grafiği dosyası bulunamadı.")
            
    with col_g2:
        if os.path.exists(karma_yolu):
            st.image(karma_yolu, caption='Karmaşıklık Matrisi (Confusion Matrix)', use_container_width=True)
        else:
            st.warning("Matris dosyası bulunamadı.")

st.divider()

# --- 7. ANALİZ ALANI ---
st.subheader("Görüntü Analizi")
col_input, col_output = st.columns([1, 1])

with col_input:
    st.write("Analiz edilecek ultrason görüntüsünü yükleyiniz:")
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Yüklenen Görüntü', use_container_width=True)

with col_output:
    st.write("Sistem Analiz Sonucu:")
    if file and model:
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Teşhisi Başlat"):
            with st.spinner('Yapay zeka analizini gerçekleştiriyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                guven = np.max(preds) * 100
                
                # Sonuç ekranı
                st.metric(label="Tahmin", value=classes[res_idx])
                st.write(f"**Güven Oranı:** %{guven:.2f}")
                st.progress(int(guven))
                
                if res_idx == 1:
                    st.error("Yüksek Risk Grubu: İleri klinik tetkik önerilir.")
                else:
                    st.success("Düşük Risk Grubu: Bulgular stabil görünmektedir.")

# --- 8. KAYNAKÇA ---
st.divider()
with st.expander("Kaynakça"):
    st.caption("1. Al-Dhabyani, W., et al. (2020). Dataset of breast ultrasound images.")
    st.caption("2. Chollet, F. (2017). Deep Learning with Python.")
    st.caption("3. Giresun Üniversitesi Bitirme Projesi Kılavuzu.")

st.caption("© 2026 Sağlık46 | Giresun Üniversitesi")
