import streamlit as st
from google import genai
from PIL import Image
import pandas as pd
import numpy as np
import json, time, re
from catboost import CatBoostRegressor

st.set_page_config(page_title="AI Araç Değerleme", page_icon="🚗", layout="centered")

# ---- Özel görünüm (koyu + mavi, modern) ----
st.markdown("""
<style>
/* Ana başlık: mavi gradient */
h1 {
    background: linear-gradient(90deg, #3B82F6, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}
/* Bölüm başlıkları için sol mavi çizgi */
h2 {
    border-left: 4px solid #3B82F6;
    padding-left: 12px;
    margin-top: 1.5rem !important;
}
/* Butonlar: yuvarlak, gradient, hover efektli */
.stButton > button {
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(59,130,246,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(59,130,246,0.5);
}
/* Giriş kutuları ve menüler: yuvarlak köşe, ince mavi çerçeve */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput div[data-baseweb="input"] {
    border-radius: 8px;
    border-color: #2A3B52 !important;
}
/* Başarı/uyarı kutuları: daha yumuşak köşe */
.stAlert { border-radius: 10px; }
/* Dosya yükleme alanı */
.stFileUploader { border-radius: 10px; }
/* Metrikleri kart gibi göster */
div[data-testid="stMetric"] {
    background: #1A2332;
    border: 1px solid #2A3B52;
    border-radius: 12px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

KATEGORIK = ["konum", "marka", "seri", "model", "vites_tipi", "yakit_tipi", "kasa_tipi", "cekis"]
SAYISAL   = ["yil", "kilometre", "motor_hacmi", "motor_gucu", "tramer", "degisen", "boyali"]

@st.cache_resource
def veri_ve_model():
    df = pd.read_csv("cars1.csv")
    df = df[KATEGORIK + SAYISAL + ["fiyat"]].dropna()
    df = df[(df["fiyat"] > 50000) & (df["fiyat"] < 15000000)]  # mantik siniri
    X = df[KATEGORIK + SAYISAL]
    y = np.log1p(df["fiyat"])  # log donusumu: carpik fiyati duzeltir
    model = CatBoostRegressor(iterations=600, depth=8, learning_rate=0.08,
                              cat_features=KATEGORIK, verbose=0, random_seed=42)
    model.fit(X, y)
    return df, model

df, model = veri_ve_model()

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("API anahtarı bulunamadı. Streamlit 'Secrets' bölümüne GEMINI_API_KEY eklemelisin.")
    st.stop()

MODELLER = ["gemini-flash-latest", "gemini-3-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-flash-lite-latest"]

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
def tr(s):
    s = str(s).lower()
    for a, b in [("ç","c"),("ş","s"),("ğ","g"),("ü","u"),("ö","o"),("ı","i"),("İ","i")]:
        s = s.replace(a, b)
    return s
def esle(secenekler, deger):
    """AI degerini menu secenegine esle (Turkce/harf/kismi tolere)."""
    if not deger: return 0
    d = tr(deger)
    for i, o in enumerate(secenekler):
        if tr(o) == d: return i
    for i, o in enumerate(secenekler):
        so = tr(o)
        if d in so or so in d: return i
    return 0
def sayi(deger, varsayilan):
    if deger is None: return varsayilan
    s = re.findall(r"\d+", str(deger))
    return int(s[0]) if s else varsayilan
def yil_cikar(m, v=2018):
    s = [int(x) for x in re.findall(r"\d{4}", str(m or ""))]
    return (s[0]+s[1])//2 if len(s) >= 2 else (s[0] if s else v)

def aralik_tahmin(arac_df):
    """CatBoost'un yaprak dagilimindan degil; agac-agac tahminlerinden aralik."""
    log_tah = model.predict(arac_df, prediction_type="RawFormulaVal")
    # Tek deger doner; belirsizlik icin benzer ilanlardan std kullanacagiz (asagida)
    return np.expm1(log_tah[0])

st.title("🚗 AI Araç Değerleme Asistanı")
st.write("Bir araç fotoğrafı yükle; yapay zeka aracın tüm özelliklerini tanısın, sen sadece yıl/km/vites gibi bilgileri onayla.")

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
                '{"marka":"","seri":"","model":"","renk":"","kasa_tipi":"","cekis":"",'
                '"motor_hacmi_cc":"","motor_gucu_bg":"","emin_olunmayanlar":"",'
                '"gorunur_durum":"","hasar_var_mi":"","tahmini_yil_araligi":"","notlar":""}\n'
                "KURALLAR:\n"
                "- motor_hacmi_cc: sadece SAYI yaz, cc cinsinden (ör. 1400, 1600). '1.4' YAZMA, 1400 yaz.\n"
                "- motor_gucu_bg: sadece SAYI yaz, beygir cinsinden (ör. 95, 130).\n"
                "- cekis: 'Önden Çekiş', 'Arkadan İtiş' veya '4WD' yaz.\n"
                "- Bu teknik bilgileri aracın marka/modelinden çıkar.\n"
                "- ÖNEMLİ DÜRÜSTLÜK KURALI: Donanım paketi (Joy/Icon/Touch gibi), motor hacmi, "
                "motor gücü gibi bilgiler fotoğraftan KESİN anlaşılmıyorsa, en olası değeri yaz AMA "
                "'emin_olunmayanlar' alanına hangi bilgilerden emin olmadığını yaz "
                "(ör. 'donanım paketi ve motor gücü fotoğraftan kesinleştirilemedi'). "
                "Emin olduğun bir şey yoksa emin_olunmayanlar'ı boş bırak. Asla uydurma, tahmin ettiğini belli et.\n"
                "- gorunur_durum: sadece gözle görülen dış durum. hasar_var_mi: 'evet' veya 'hayir'.\n"
                "- Kesin ekspertiz (değişen/boyalı parça) yorumu YAPMA.\n"
                "Türkçe cevap ver."
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
    st.success("Analiz tamamlandı — özellikler otomatik dolduruldu. Aşağıda yıl/km/vites bilgilerini onayla.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Marka", analiz.get("marka", "-"))
    c2.metric("Model", analiz.get("model", "-"))
    c3.metric("Renk", analiz.get("renk", "-"))
    st.write(f"**Görünür durum:** {analiz.get('gorunur_durum','-')}")
    if analiz.get("notlar"):
        st.info("📝 " + analiz["notlar"])
    # AI'in emin olmadigi bilgiler icin uyari
    emin_degil = analiz.get("emin_olunmayanlar", "")
    if emin_degil and str(emin_degil).lower() not in ("", "belirsiz", "yok", "-"):
        st.warning(f"🤔 Yapay zeka şunlardan emin değil, lütfen aşağıdan kontrol et: {emin_degil}")
    if str(analiz.get("hasar_var_mi", "")).lower() == "evet":
        st.warning("⚠️ Fotoğrafta görünür hasar tespit edildi. Aşağıdan hasar bilgilerini girersen tahmin daha doğru olur.")
    st.caption("⚠️ Bu bir ön izlenimdir, sadece fotoğrafa dayanır ve gerçek ekspertiz yerine geçmez.")

# ============ BÖLÜM 2: FİYAT TAHMİNİ ============
st.header("2) Fiyat Tahmini")

# AI'dan gelen on-degerler
a = analiz or {}
on_yil = yil_cikar(a.get("tahmini_yil_araligi")) if analiz else 2018

markalar = sirala(df["marka"])
marka = st.selectbox("Marka", markalar, index=esle(markalar, a.get("marka")))
seri_opt = sirala(df[df["marka"] == marka]["seri"])
seri = st.selectbox("Seri", seri_opt, index=esle(seri_opt, a.get("seri") or a.get("model")))
model_opt = sirala(df[(df["marka"] == marka) & (df["seri"] == seri)]["model"])
secili_model = st.selectbox("Model", model_opt, index=esle(model_opt, a.get("model")))

# Secilen araca ait alt kume: menuleri buna gore filtreleyecegiz
alt_kume = df[(df["marka"] == marka) & (df["seri"] == seri)]
if len(alt_kume) == 0:
    alt_kume = df[df["marka"] == marka]

def akilli_alan(kap, etiket, sutun, ai_deger):
    """Araçta tek seçenek varsa otomatik kullan+göster; çoksa menü çıkar."""
    secenekler = sirala(alt_kume[sutun])
    if not secenekler:
        secenekler = sirala(df[sutun])  # bos kalirsa tum secenekler
    if len(secenekler) == 1:
        kap.markdown(f"**{etiket}:** {secenekler[0]}  \n<span style='color:gray;font-size:0.8em'>(bu araçta tek seçenek)</span>", unsafe_allow_html=True)
        return secenekler[0]
    return kap.selectbox(etiket, secenekler, index=esle(secenekler, ai_deger))

st.markdown("**Senin gireceklerin** (yıl, kilometre, vites, şehir):")
col1, col2 = st.columns(2)
yil = col1.number_input("Model yılı", min_value=1990, max_value=2026, value=int(on_yil))
kilometre = col2.number_input("Kilometre", min_value=0, max_value=1000000, value=100000, step=5000)
col1b, col2b = st.columns(2)
# Vites: tek seçenekse otomatik, çoksa menü
vites_tipi = akilli_alan(col1b, "Vites tipi", "vites_tipi", a.get("vites_tipi"))
konum = col2b.selectbox("Şehir", sirala(df["konum"]))

with st.expander("AI'ın doldurduğu diğer bilgiler (gerekirse düzelt)"):
    col5, col6 = st.columns(2)
    yakit_tipi = akilli_alan(col5, "Yakıt tipi", "yakit_tipi", a.get("yakit_tipi"))
    kasa_tipi  = akilli_alan(col6, "Kasa tipi", "kasa_tipi", a.get("kasa_tipi"))
    col7, col8, col9 = st.columns(3)
    cekis = akilli_alan(col7, "Çekiş", "cekis", a.get("cekis"))
    motor_hacmi = col8.number_input("Motor hacmi (cc)", min_value=600, max_value=8000,
                                    value=sayi(a.get("motor_hacmi_cc"), 1600), step=100)
    motor_gucu = col9.number_input("Motor gücü (bg)", min_value=30, max_value=1000,
                                   value=sayi(a.get("motor_gucu_bg"), 120), step=10)

ai_hasar = bool(analiz and str(a.get("hasar_var_mi", "")).lower() == "evet")
hasar_var = st.checkbox("Araçta hasar / tramer kaydı var mı?", value=ai_hasar)
if hasar_var:
    col10, col11, col12 = st.columns(3)
    tramer  = col10.number_input("Tramer kaydı (TL)", min_value=0, max_value=5000000, value=20000, step=5000)
    degisen = col11.number_input("Değişen parça sayısı", min_value=0, max_value=30, value=1)
    boyali  = col12.number_input("Boyalı parça sayısı", min_value=0, max_value=30, value=1)
else:
    tramer, degisen, boyali = 0, 0, 0

if st.button("Fiyatı Tahmin Et", type="primary"):
    arac = pd.DataFrame([{
        "konum": konum, "marka": marka, "seri": seri, "model": secili_model,
        "vites_tipi": vites_tipi, "yakit_tipi": yakit_tipi, "kasa_tipi": kasa_tipi,
        "cekis": cekis, "yil": yil, "kilometre": kilometre,
        "motor_hacmi": motor_hacmi, "motor_gucu": motor_gucu,
        "tramer": tramer, "degisen": degisen, "boyali": boyali}])

    tahmin = aralik_tahmin(arac)
    # Belirsizlik: veride benzer araclarin fiyat dagilimindan aralik
    benzer = df[(df["marka"] == marka) & (df["seri"] == seri)]
    n_benzer = len(benzer)
    if n_benzer >= 5:
        std = benzer["fiyat"].std()
        alt, ust = max(0, tahmin - std), tahmin + std
    else:
        alt, ust = tahmin * 0.85, tahmin * 1.15

    st.success(f"Tahmini fiyat: {tahmin:,.0f} TL")
    st.write(f"**Tahmini aralık:** {alt:,.0f} TL – {ust:,.0f} TL")
    st.caption(f"Bu tahmin, veri setindeki **{n_benzer}** benzer ilana dayanmaktadır.")
    if n_benzer < 15:
        st.warning(f"⚠️ Bu araç ({marka} {seri}) için veride yalnızca {n_benzer} ilan var. "
                   "Tahmin daha az kesin olabilir.")
    if n_benzer > 0:
        st.write("**Veri setindeki benzer araçlar (gerçek fiyatlar):**")
        ornekler = benzer.copy()
        ornekler["yil_farki"] = (ornekler["yil"] - yil).abs()
        gosterim = ornekler.sort_values("yil_farki").head(5)[["yil", "kilometre", "fiyat"]]
        gosterim = gosterim.rename(columns={"yil": "Yıl", "kilometre": "Kilometre", "fiyat": "Fiyat (TL)"})
        st.dataframe(gosterim.style.format({"Kilometre": "{:,.0f}", "Fiyat (TL)": "{:,.0f}"}),
                     hide_index=True, use_container_width=True)
    hasar_notu = " (hasarsız)" if (tramer == 0 and degisen == 0 and boyali == 0) else " (hasar dahil)"
    st.caption(f"{marka} {seri} • {yil} • {kilometre:,.0f} km{hasar_notu}")
