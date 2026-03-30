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

# --- 4. AKADEMİK ÜST BAŞLIK ---
st.header("Sağlık46: Meme Kanseri Teşhis Sistemi")
st.write(f"**Araştırmacı:** Emine Berk (2207060044) | **Danışman:** Dr. Öğr. Üyesi Muhammet Çakmak | Giresun Üniversitesi")
st.divider()

# --- 5. DETAYLI PROJE ÖZETİ VE CNN METODOLOJİSİ ---
with st.expander("Proje Detayları ve Derin Öğrenme Metodolojisi", expanded=True):
    st.write("### Proje Hakkında")
    st.write("""
    Bu proje, meme kanserinin erken teşhisinde radyologlara destek olmak amacıyla geliştirilmiş bir **Bilgisayar Destekli Tanı (CAD)** sistemidir. 
    Derin öğrenme mimarilerinden olan **CNN (Convolutional Neural Networks)** kullanılarak, ultrason görüntülerindeki mikroskobik doku değişimleri analiz edilmektedir.
    """)
    
    col_tech1, col_tech2 = st.columns(2)
    with col_tech1:
        st.write("### Teknik Katman Analizi")
        st.info("""
        * **Convolutional (Evrişim) Katmanı:** Görüntü üzerindeki pikselleri tarayarak kenar, köşe ve doku (tümörün sınır hatları gibi) özelliklerini çıkarır.
        * **Pooling (Havuzlama):** Öznitelik haritasını küçülterek hesaplama maliyetini düşürür ve en belirgin özellikleri (maksimum sinyal) ön plana çıkarır.
        * **Dropout:** Modelin sadece eğitim verisini ezberlemesini (overfitting) önlemek için bazı nöronları rastgele devre dışı bırakır.
        """)
        
    with col_tech2:
        st.write("### Sınıflandırma Mantığı")
        st.success("""
        Modelimiz, 128x128 çözünürlüğe indirgenmiş ultrason verilerini şu üç kategoride sınıflandırır:
        1. **Normal:** Herhangi bir kitle veya patolojik bulgu izlenmeyen doku.
        2. **Benign (İyi Huylu):** Düzenli sınırlara sahip, genellikle tehlikesiz kitleler.
        3. **Malignant (Kötü Huylu):** Düzensiz sınırlı, yayılma eğilimi gösteren yüksek riskli kitleler.
        """)

st.divider()

# --- 6. EĞİTİM GRAFİKLERİ (YAN YANA VE DOĞRU SIRALAMA) ---
st.write("### Model Eğitim Performans Verileri")
col_g1, col_g2 = st.columns(2)

# GitHub'daki GERÇEK dosya isimlerin
grafik_yolu = 'Ekran görüntüsü 2026-03-29 231910.png' # Accuracy/Loss
karma_yolu = 'Ekran görüntüsü 2026-03-29 232001.png'   # Confusion Matrix

with col_g1:
    st.write("**Öğrenme Eğrileri (Accuracy & Loss)**")
    if os.path.exists(grafik_yolu):
        st.image(grafik_yolu, use_container_width=True, caption="Modelin eğitim sürecindeki başarı ve kayıp grafiği.")
    else:
        st.warning("Grafik dosyası bulunamadı.")

with col_g2:
    st.write("**Hata Analizi (Confusion Matrix)**")
    if os.path.exists(karma_yolu):
        st.image(karma_yolu, use_container_width=True, caption="Gerçek değerler ile tahmin edilen değerlerin karşılaştırma matrisi.")
    else:
        st.warning("Matris dosyası bulunamadı.")

st.divider()

# --- 7. ANALİZ ALANI ---
st.subheader("Görüntü Analizi ve Canlı Teşhis")
c1, c2 = st.columns([1, 1])

with c1:
    st.write("**Veri Girişi**")
    file = st.file_uploader("Analiz için ultrason görüntüsü yükleyiniz (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Sisteme Aktarılan Görüntü', use_container_width=True)

with c2:
    st.write("**Yapay Zeka Karar Mekanizması**")
    if file and model:
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Analizi Başlat"):
            with st.spinner('Pikseller işleniyor ve katmanlar arası analiz yapılıyor...'):
                preds = model.predict(img_array)
                classes = ['İyi Huylu (Benign)', 'Kötü Huylu (Malignant)', 'Normal']
                res_idx = np.argmax(preds)
                guven = np.max(preds) * 100
                
                st.metric("Tahmin Edilen Sınıf", classes[res_idx])
                st.write(f"**Güven Oranı (Confidence Score):** %{guven:.2f}")
                st.progress(int(guven))
                
                if res_idx == 1:
                    st.error("Kritik Uyarı: Malignant (Kötü Huylu) bulgu tespit edildi. Acil klinik biyopsi ve uzman görüşü önerilir.")
                else:
                    st.success("Düşük Risk: Mevcut veriler ışığında normal/iyi huylu doku yapısı gözlemlenmiştir.")

# --- 8. AKADEMİK KAYNAKÇA ---
st.divider()
with st.expander("Akademik Referanslar"):
    st.caption("1. Al-Dhabyani, W., et al. (2020). Dataset of breast ultrasound images. Data in Brief.")
    st.caption("2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.")
    st.caption("3. Esteva, A., et al. (2017). Dermatologist-level classification with deep neural networks. Nature.")

st.caption("© 2026 Sağlık46 Projesi | Giresun Üniversitesi Veri Bilimi Çalışması")
