import streamlit as st
from google import genai
from PIL import Image
import pandas as pd
import json, time, re
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI Araç Değerleme", page_icon="🚗")

KATEGORIK = ["konum", "marka", "seri", "model", "vites_tipi", "yakit_tipi", "kasa_tipi", "cekis"]
# Hasar bilgileri de artik ozellik: tramer, degisen, boyali
SAYISAL   = ["yil", "kilometre", "motor_hacmi", "motor_gucu", "tramer", "degisen", "boyali"]

@st.cache_resource
def veri_ve_model():
    df = pd.read_csv("cars1.csv")
    df = df[KATEGORIK + SAYISAL + ["fiyat"]].dropna()
    df = df[(df["fiyat"] > 50000) & (df["fiyat"] < 15000000)]
    X, y = df[KATEGORIK + SAYISAL], df["fiyat"]
    onisleme = ColumnTransformer(
        [("kat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), KATEGORIK)],
        remainder="passthrough")
    pipe = Pipeline([("onisleme", onisleme),
        ("model", RandomForestRegressor(n_estimators=40, max_depth=18, random_state=42, n_jobs=-1))])
    pipe.fit(X, y)
    return df, pipe

df, pipe = veri_ve_model()

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("API anahtarı bulunamadı. Streamlit 'Secrets' bölümüne GEMINI_API_KEY eklemelisin.")
    st.stop()

MODELLER = ["gemini-3-flash", "gemini-2.5-flash", "gemini-flash-latest"]

def analiz_et(komut, resim):
    son_hata = None
    for model_adi in MODELLER:
        for _ in range(3):
            try:
                return client.models.generate_content(model=model_adi, contents=[komut, resim]).text.strip()
            except Exception as e:
                son_hata = e
                if any(k in str(e) for k in ["503", "UNAVAILABLE", "429"]):
                    time.sleep(2); continue
                else:
                    break
    raise son_hata

def sirala(seri): return sorted(seri.dropna().astype(str).unique().tolist())
def yil_cikar(m, v=2018):
    s = [int(x) for x in re.findall(r"\d{4}", str(m or ""))]
    return (s[0]+s[1])//2 if len(s) >= 2 else (s[0] if s else v)
def idx(secenekler, deger):
    if deger:
        for i, o in enumerate(secenekler):
            if str(o).lower() == str(deger).lower(): return i
    return 0

st.title("🚗 AI Araç Değerleme Asistanı")
st.write("Bir araç fotoğrafı yükle; yapay zeka aracı ve durumunu tanısın, sonra tahmini fiyatını hesaplayalım.")

# ============ BÖLÜM 1: FOTOĞRAF ANALİZİ ============
st.header("1) Fotoğraf Analizi")
yuklenen = st.file_uploader("Araç fotoğrafı yükle", type=["jpg", "jpeg", "png"])

if yuklenen is not None:
    resim = Image.open(yuklenen)
    st.image(resim, caption="Yüklenen fotoğraf", use_container_width=True)
    if st.button("Fotoğrafı Analiz Et", type="primary"):
        with st.spinner("Yapay zeka fotoğrafı inceliyor..."):
            komut = (
                "Bu bir araç fotoğrafı. SADECE şu JSON formatında cevap ver, başka metin ekleme:\n"
                '{"marka":"","model":"","renk":"","kasa_tipi":"","gorunur_durum":"",'
                '"hasar_var_mi":"","tahmini_yil_araligi":"","notlar":""}\n'
                "ÖNEMLİ: Sadece fotoğrafta GÖRÜNEN şeyleri söyle. gorunur_durum = gözle görülebilen dış durum. "
                "hasar_var_mi = sadece 'evet' veya 'hayir' yaz (gözle görülür hasar var mı). "
                "Değişen/boyalı parça gibi kesin ekspertiz yorumu YAPMA. "
                "Emin değilsen 'belirsiz' yaz, asla uydurma. Türkçe cevap ver."
            )
            try:
                metin = analiz_et(komut, resim)
                if "```" in metin:
                    metin = metin.split("```")[1].replace("json", "", 1).strip()
                st.session_state["analiz"] = json.loads(metin)
            except Exception as e:
                st.error(f"Analiz hatası: {e}")

analiz = st.session_state.get("analiz")
if analiz:
    st.success("Analiz tamamlandı — aşağıdaki bilgiler otomatik dolduruldu, kontrol edip düzeltebilirsin.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Marka", analiz.get("marka", "-"))
    c2.metric("Model", analiz.get("model", "-"))
    c3.metric("Renk", analiz.get("renk", "-"))
    st.write(f"**Görünür durum:** {analiz.get('gorunur_durum','-')}")
    if analiz.get("notlar"):
        st.info("📝 " + analiz["notlar"])
    if str(analiz.get("hasar_var_mi", "")).lower() == "evet":
        st.warning("⚠️ Fotoğrafta görünür hasar tespit edildi. Aşağıdaki hasar bilgilerini "
                   "(ekspertiz raporundan) girersen fiyat tahmini daha doğru olur.")
    st.caption("⚠️ Bu bir ön izlenimdir, sadece fotoğrafa dayanır ve gerçek ekspertiz yerine geçmez.")

# ============ BÖLÜM 2: FİYAT TAHMİNİ ============
st.header("2) Fiyat Tahmini")
st.write("Fotoğraftan gelen bilgiler dolduruldu. Kesin bilgileri (yıl, km, hasar) sen tamamla.")

on_marka = analiz.get("marka") if analiz else None
on_seri  = analiz.get("model") if analiz else None
on_yil   = yil_cikar(analiz.get("tahmini_yil_araligi")) if analiz else 2018

col1, col2 = st.columns(2)
with col1:
    konum = st.selectbox("Şehir", sirala(df["konum"]))
    markalar = sirala(df["marka"])
    marka = st.selectbox("Marka", markalar, index=idx(markalar, on_marka))
    seri_opt = sirala(df[df["marka"] == marka]["seri"])
    seri = st.selectbox("Seri", seri_opt, index=idx(seri_opt, on_seri))
    model_opt = sirala(df[(df["marka"] == marka) & (df["seri"] == seri)]["model"])
    secili_model = st.selectbox("Model", model_opt)
    yakit_tipi = st.selectbox("Yakıt tipi", sirala(df["yakit_tipi"]))
with col2:
    vites_tipi = st.selectbox("Vites tipi", sirala(df["vites_tipi"]))
    kasa_tipi  = st.selectbox("Kasa tipi", sirala(df["kasa_tipi"]))
    cekis      = st.selectbox("Çekiş", sirala(df["cekis"]))
    yil        = st.number_input("Model yılı", 1990, 2026, value=int(on_yil))
    kilometre  = st.number_input("Kilometre", 0, value=100000, step=5000)

st.subheader("Motor Bilgileri")
col3, col4 = st.columns(2)
motor_hacmi = col3.number_input("Motor hacmi (cc)", 800, value=1600, step=100)
motor_gucu  = col4.number_input("Motor gücü (bg)", 40, value=120, step=10)

# AI fotoğrafta hasar gördüyse kutu otomatik işaretli gelir
ai_hasar = bool(analiz and str(analiz.get("hasar_var_mi", "")).lower() == "evet")
hasar_var = st.checkbox("Araçta hasar / tramer kaydı var mı?", value=ai_hasar)

if hasar_var:
    st.write("Ekspertiz bilgilerini gir (bilmiyorsan tahmini değer girebilirsin):")
    col5, col6, col7 = st.columns(3)
    tramer  = col5.number_input("Tramer kaydı (TL)", 0, value=20000, step=5000,
                                help="Araçtaki toplam hasar kayıt tutarı.")
    degisen = col6.number_input("Değişen parça sayısı", 0, 30, value=1)
    boyali  = col7.number_input("Boyalı parça sayısı", 0, 30, value=1)
else:
    tramer, degisen, boyali = 0, 0, 0  # hasarsız

if st.button("Fiyatı Tahmin Et", type="primary"):
    arac = pd.DataFrame([{
        "konum": konum, "marka": marka, "seri": seri, "model": secili_model,
        "vites_tipi": vites_tipi, "yakit_tipi": yakit_tipi, "kasa_tipi": kasa_tipi,
        "cekis": cekis, "yil": yil, "kilometre": kilometre,
        "motor_hacmi": motor_hacmi, "motor_gucu": motor_gucu,
        "tramer": tramer, "degisen": degisen, "boyali": boyali}])
    tahmin = pipe.predict(arac)[0]
    st.success(f"Tahmini fiyat: {tahmin:,.0f} TL")
    hasar_notu = " (hasarsız)" if (tramer == 0 and degisen == 0 and boyali == 0) else " (hasar dahil edildi)"
    st.caption(f"{marka} {seri} • {yil} • {kilometre:,.0f} km{hasar_notu}")
