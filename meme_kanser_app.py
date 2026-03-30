import streamlit as st
import os
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf

# --- 1. SAYFA AYARLARI VE TASARIMI (CSS) ---
st.set_page_config(page_title="Sağlık46 | Giresun Üni.", layout="wide")

# Grafikleri yan yana getirmek ve tasarımı güzelleştirmek için özel CSS
st.markdown("""
    <style>
    /* Başlık ve Bilgileri Ortala */
    .title-text {
        text-align: center;
        color: #FFFFFF;
        font-family: 'Times New Roman', Times, serif;
    }
    .info-text {
        text-align: center;
        color: #CCCCCC;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    /* Expander Başlıklarını ve İçeriğini Güzelleştir */
    .streamlit-expanderHeader {
        background-color: #1A1A1A;
        border-radius: 5px;
        color: #FFFFFF !important;
        font-size: 1.2rem;
        font-family: serif;
    }
    .streamlit-expanderContent {
        background-color: #262626;
        padding: 20px;
        border-radius: 0 0 5px 5px;
        border: 1px solid #333;
    }
    
    /* Görselleri ve Analiz Kısımlarını Çerçevele */
    .image-frame {
        border: 2px solid #333;
        border-radius: 8px;
        padding: 10px;
        background-color: #1A1A1A;
    }
    .result-box {
        border: 2px solid #444;
        border-radius: 8px;
        padding: 20px;
        background-color: #262626;
        margin-top: 15px;
    }
    
    /* Metrik Başlığını Güzelleştir */
    [data-testid="stMetricValue"] {
        font-family: serif;
        font-weight: bold;
    }
    </style>
""", unsafe_mode=True)

# --- 2. HATA GİDERİCİ (CUSTOM OBJECTS) ---
class FixedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# --- 3. MODEL YÜKLEME VE DRIVE'DAN İNDİRME ---
@st.cache_resource
def model_getir():
    model_yolu = 'Meme_Kanseri_Final_Modeli.h5'
    if not os.path.exists(model_yolu):
        with st.spinner('Model dosyası sunucuya indiriliyor, lütfen bekleyiniz...'):
            file_id = '14OW6zCuzug3Ge7dZoqPuKrbFuOXr6meb'
            drive_url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(drive_url, model_yolu, quiet=False)
            except Exception as e:
                st.error(f"İndirme Hatası: {e}")
                return None
    
    try:
        # custom_objects ile Dense hatasını, safe_mode=False ile versiyon farkını çözüyoruz
        return tf.keras.models.load_model(
            model_yolu, 
            custom_objects={'Dense': FixedDense}, 
            compile=False, 
            safe_mode=False
        )
    except Exception as e:
        st.error(f"Model Yükleme Hatası: {e}")
        return None

model = model_getir()

# --- 4. ÜST BAŞLIK VE AKADEMİK BİLGİLER ---
st.markdown("<h1 class='title-text'>Sağlık46: Meme Kanseri Teşhis Sistemi</h1>", unsafe_mode=True)
st.markdown(f"<p class='info-text'>Giresun Üniversitesi Mühendislik Fakültesi<br>Araştırmacı: Emine Berk (2207060044)<br>Danışman: Dr. Öğr. Üyesi Muhammet Çakmak</p>", unsafe_mode=True)
st.divider()

# --- 5. ÖZET VE PERFORMANS (YAN YANA KOLONLAR) ---
col_ozet, col_grafik = st.columns([1, 1])

with col_ozet:
    with st.expander("📖 Proje Özeti ve Metodoloji", expanded=False):
        st.write("""
        ### Proje Amacı
        Bu çalışma, evrişimli sinir ağları (CNN) mimarisi kullanılarak ultrason görüntülerinden meme kanseri teşhisine yardımcı karar destek mekanizması geliştirilmesi üzerinedir.
        
        ### Veri Seti ve Sınıflar
        Model; Normal, İyi Huylu (Benign) ve Kötü Huylu (Malignant) olmak üzere üç temel sınıfta eğitilmiştir.
        
        ### Yasal Uyarı
        Bu sistem bir karar destek aracıdır. Klinik teşhis yerine geçmez ve kesin teşhis için bir radyolog onayına ihtiyaç duyar.
        """)

with col_grafik:
    with st.expander("📊 Model Eğitim Performansı (Grafikler)", expanded=True):
        st.write("Modelin eğitim sürecindeki başarı (Accuracy) ve kayıp (Loss) grafikleri ile Karmaşıklık Matrisi (Confusion Matrix) aşağıdadır.")
        
        # GitHub'daki GERÇEK dosya isimlerin bunlardır ( image_967ebe.png'den kontrol edildi):
        grafik_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' 
        karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'
        
        col_a, col_b = st.columns(2)
        
        # Grafik 1: Accuracy/Loss
        if os.path.exists(grafik_yolu):
            col_a.image(grafik_yolu, caption="Eğitim Başarı ve Kayıp Grafiği", use_container_width=True)
        else:
            col_a.error(f"Hata: {grafik_yolu} bulunamadı.")
            
        # Grafik 2: Confusion Matrix
        if os.path.exists(karma_yolu):
            col_b.image(karma_yolu, caption="Karmaşıklık Matrisi (Confusion Matrix)", use_container_width=True)
        else:
            col_b.error(f"Hata: {karma_yolu} bulunamadı.")

st.divider()

# --- 6. ANALİZ ALANI ---
st.subheader("🔬 Ultrason Görüntü Analizi")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='image-frame'>", unsafe_mode=True)
    st.write("**Görüntü Yükleme**")
    file = st.file_uploader("Ultrason Görüntüsü Yükleyin (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Analiz İçin Yüklenen Görüntü', use_container_width=True)
    st.markdown("</div>", unsafe_mode=True)

with col2:
    if file and model:
        st.markdown("<div class='result-box'>", unsafe_mode=True)
        st.write("**Teşhis Sonuç Raporu**")
        
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("🔎 Analizi Başlat"):
            with st.spinner('Yapay zeka analiz ediyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                oran = np.max(preds) * 100
                
                st.metric("Tahmin Edilen Sınıf", classes[res_idx])
                st.write(f"**Güven Oranı:** %{oran:.2f}")
                st.progress(int(oran))
                
                if res_idx == 1:
                    st.error("DİKKAT: Yüksek riskli bulgu tespit edildi. Acil uzman hekim incelemesi önerilir.")
                elif res_idx == 0:
                    st.info("ℹ️ İyi huylu doku benzerliği tespit edildi. Takip önerilir.")
                else:
                    st.success("Bulgular düşük risk grubunda değerlendirilmiştir.")
        st.markdown("</div>", unsafe_mode=True)

# --- 7. KAYNAKÇA ---
st.divider()
with st.expander("📚 Akademik Kaynakça"):
    st.write("""
    1. Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images. Data in Brief.
    2. Chollet, F. (2017). Deep Learning with Python. Manning Publications.
    3. Giresun Üniversitesi Bitirme Projesi Kılavuzu.
    """)

st.caption("© 2026 Sağlık46 Projesi - Giresun Üniversitesi")
