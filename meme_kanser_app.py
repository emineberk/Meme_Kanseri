import streamlit as st
import os
import numpy as np
from PIL import Image

# Sayfa Ayarları
st.set_page_config(page_title="Sağlık46 | Emine Berk", page_icon="🏥")

# Başlıklar
st.title("🏥 python -m streamlit run meme_kanser_app.pySağlık46: Meme Kanseri Teşhis Sistemi")
st.info("Geliştiren: **Emine Berk** | Danışman: **Dr. Öğr. Üyesi Muhammet Çakmak**")

@st.cache_resource
def model_getir():
    try:
        # Keras'ı bağımsız çağırmak yerine doğrudan tf içinden alıyoruz
        import tensorflow as tf
        model_yolu = 'Meme_Kanseri_Final_Modeli.h5'
        
        if os.path.exists(model_yolu):
            # En güncel ve hatasız yükleme yöntemi budur
            return tf.keras.models.load_model(model_yolu, compile=False)
        else:
            st.error(f"Dosya bulunamadı: {model_yolu}")
            return None
    except Exception as e:
        st.error(f"Sistemsel bir hata oluştu: {e}")
        return None

model = model_getir()

if model:
    st.sidebar.success("✅ Sistem Analize Hazır!")
else:
    st.sidebar.error("❌ Model Yüklenemedi")

# Dosya Yükleme
file = st.file_uploader("Ultrason Görüntüsü Seçin", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, caption='Yüklenen Görüntü', use_container_width=True)
    
    # Modelin beklediği 128x128 boyutuna getiriyoruz
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Analiz Et"):
        with st.spinner('Yapay zeka analiz ediyor...'):
            preds = model.predict(img_array)
            classes = ['Benign (İyi Huylu)', 'Malignant (Kötü Huylu)', 'Normal']
            sonuc = classes[np.argmax(preds)]
            oran = np.max(preds) * 100
            
            st.markdown("---")
            st.subheader(f"Tahmin Sonucu: **{sonuc}**")
            st.write(f"Güven Oranı: %{oran:.2f}")
            st.progress(int(oran))
            st.warning("Bu bir yardımcı tanı aracıdır. Kesin teşhis için doktor onayı gereklidir.")