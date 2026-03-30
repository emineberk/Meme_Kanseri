import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown  # Drive'dan indirmek için şart!

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Sağlık46 | Emine Berk", page_icon="🏥")

# --- 2. BAŞLIK VE AKADEMİK BİLGİLER ---
st.title("🩺 Sağlık46: Meme Kanseri Teşhis Sistemi")
st.info("Giresun Üniversitesi Mühendislik Fakültesi | **Araştırmacı:** Emine Berk (2207060044) | **Danışman:** Dr. Öğr. Üyesi Muhammet Çakmak")

# --- 3. MODELİ DRIVE'DAN İNDİRME VE YÜKLEME ---
@st.cache_resource
def model_getir():
    model_yolu = 'Meme_Kanseri_Final_Modeli.h5'
    
    # Dosya sunucuda yoksa Drive'dan indir
    if not os.path.exists(model_yolu):
        with st.spinner('Model dosyası buluttan çekiliyor, lütfen bekleyiniz...'):
            file_id = '14OW6zCuzug3Ge7dZoqPuKrbFuOXr6meb'
            drive_url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(drive_url, model_yolu, quiet=False)
            except Exception as e:
                st.error(f"Model indirilirken hata oluştu: {e}")
                return None

    # Modeli TensorFlow ile yükle
    try:
        import tensorflow as tf
        if os.path.exists(model_yolu):
            return tf.keras.models.load_model(model_yolu, compile=False)
        else:
            st.error("Model dosyası indirme sonrası bulunamadı!")
            return None
    except Exception as e:
        st.error(f"Sistemsel bir hata oluştu: {e}")
        return None

# Modeli çağır
model = model_getir()

# --- 4. YAN PANEL (SIDEBAR) ---
if model:
    st.sidebar.success("✅ Sistem Analize Hazır!")
else:
    st.sidebar.error("❌ Model Yüklenemedi")
    st.sidebar.warning("Lütfen Google Drive linkinin 'Herkes'e açık olduğunu kontrol edin.")

# --- 5. ANALİZ ALANI ---
file = st.file_uploader("Analiz için Ultrason Görüntüsü Seçin", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, caption='Yüklenen Görüntü', use_container_width=True)
    
    # Modelin beklediği 128x128 boyutuna getiriyoruz
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Analizi Başlat"):
        if model is not None:
            with st.spinner('Yapay zeka analiz ediyor...'):
                preds = model.predict(img_array)
                classes = ['Benign (İyi Huylu)', 'Malignant (Kötü Huylu)', 'Normal']
                res_idx = np.argmax(preds)
                sonuc = classes[res_idx]
                oran = np.max(preds) * 100
                
                st.markdown("---")
                st.subheader(f"Teşhis Sonucu: **{sonuc}**")
                st.write(f"Güven Oranı: %{oran:.2f}")
                st.progress(int(oran))
                
                if res_idx == 1: # Malignant durumu
                    st.error("🚨 DİKKAT: Şüpheli bulgu tespit edildi. Uzman radyolog incelemesi önerilir.")
                else:
                    st.success("✅ Bulgular düşük risk grubunda değerlendirilmiştir.")
                
                st.warning("⚠️ Bu bir yardımcı karar destek aracıdır. Kesin teşhis yerine geçmez.")
        else:
            st.error("Analiz yapılamıyor: Model yüklü değil.")
