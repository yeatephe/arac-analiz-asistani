import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import json

st.set_page_config(page_title="AI Araç Analiz Asistanı", page_icon="🚗")
st.title("🚗 AI Araç Analiz Asistanı")
st.write("Bir aracın fotoğrafını yükle, yapay zeka markasını, modelini ve durumunu analiz etsin.")

# API anahtari Streamlit'in gizli "secrets" bolumunden okunur (kod icine YAZILMAZ)
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("API anahtarı bulunamadı. Streamlit ayarlarından 'Secrets' bölümüne GEMINI_API_KEY eklemelisin.")
    st.stop()

yuklenen = st.file_uploader("Araç fotoğrafı yükle", type=["jpg", "jpeg", "png"])

if yuklenen is not None:
    resim = Image.open(yuklenen)
    st.image(resim, caption="Yüklenen fotoğraf", use_container_width=True)

    if st.button("Analiz Et", type="primary"):
        with st.spinner("Yapay zeka fotoğrafı inceliyor..."):
            # Modelden JSON formatinda yapisal cevap istiyoruz
            komut = (
                "Bu bir araç fotoğrafı. Aracı incele ve SADECE şu JSON formatında cevap ver, "
                "başka hiçbir metin ekleme:\n"
                '{"marka": "", "model": "", "renk": "", "kasa_tipi": "", '
                '"durum": "", "tahmini_yil_araligi": "", "notlar": ""}\n'
                "Alanların açıklaması: durum = aracın görünür durumu (temiz, çizik, hasarlı vb.), "
                "notlar = dikkat çeken detaylar. Emin değilsen 'belirsiz' yaz. Türkçe cevap ver."
            )
            try:
                cevap = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[komut, resim],
                )
                metin = cevap.text.strip()
                # JSON'u ayikla (model bazen ```json ... ``` ile sarabilir)
                if "```" in metin:
                    metin = metin.split("```")[1].replace("json", "", 1).strip()
                veri = json.loads(metin)

                st.success("Analiz tamamlandı!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Marka", veri.get("marka", "-"))
                    st.metric("Model", veri.get("model", "-"))
                    st.metric("Renk", veri.get("renk", "-"))
                with col2:
                    st.metric("Kasa tipi", veri.get("kasa_tipi", "-"))
                    st.metric("Tahmini yıl", veri.get("tahmini_yil_araligi", "-"))
                    st.metric("Durum", veri.get("durum", "-"))
                if veri.get("notlar"):
                    st.info("📝 Notlar: " + veri["notlar"])
            except json.JSONDecodeError:
                st.warning("Model beklenmedik bir formatta cevap verdi. Ham cevap:")
                st.write(metin)
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
