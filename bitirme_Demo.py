"""
LLM Tabanlı Politika Olay Çıkarımı ve Finansal Volatilite Modellemesi
Bitirme Projesi - Gelişmiş Demo (v2)

İleride FED/TCMB raporları ile eğitilecek şekilde modüler tasarlanmıştır.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# KONFİGÜRASYON - Tek yerden kontrol
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "ticker": "XU100.IS",
    "period": "6mo",
    "interval": "1d",
    "garch_p": 1,
    "garch_q": 1,
    "seed": 42,
    "rolling_window": 20,
    "output_csv": "bitirme_sonuclar.csv",
    "output_plot": "bitirme_grafikler.png",
}

# ══════════════════════════════════════════════════════════════════════════════
# FAZ 1: VERİ TOPLAMA
# ══════════════════════════════════════════════════════════════════════════════
def veri_cek(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """yfinance ile fiyat verisi çeker, günlük yüzdesel getiri hesaplar."""
    raw = yf.download(ticker, period=period, interval=interval, progress=False)
    close = raw['Close'].dropna()
    df = pd.DataFrame({
        'Close': close.values.flatten(),
    }, index=close.index)
    df['Return'] = 100 * df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# FAZ 2: POLİTİK OLAY ÇIKARIMI (MOCK NLP)
# ══════════════════════════════════════════════════════════════════════════════

# FED ve TCMB tarzı mock haberler (ileride gerçek raporlarla değiştirilecek)
MOCK_HABERLER = {
    "tcmb": [
        "TCMB politika faizini 500 baz puan artırdı, piyasalarda tedirginlik hakim.",
        "Merkez Bankası enflasyon hedefini korudu, faiz sabit bırakıldı, piyasa stabil.",
        "TCMB döviz rezervleri eridi, kur baskısı artıyor, gerilim tırmanıyor.",
        "Para politikası kurulu toplantısında sürpriz olmadı, beklentiler yolunda.",
        "TCMB başkanı sıkılaştırma mesajı verdi, belirsizlik devam ediyor.",
        "Enflasyon raporu beklentilerin altında geldi, piyasalar olumlu karşıladı.",
        "Merkez Bankası swap ihalesi düzenledi, likidite yönetimi devam ediyor.",
    ],
    "fed": [
        "Fed faiz oranlarını 25 baz puan artırdı, küresel piyasalarda tedirginlik.",
        "Powell: Ekonomi güçlü, yumuşak iniş mümkün. Piyasalar stabil seyrediyor.",
        "ABD istihdam verileri güçlü geldi, faiz indirimi beklentisi zayıfladı.",
        "Fed bilanço küçültmeye devam, gelişen piyasalarda sermaye çıkışı riski.",
        "FOMC tutanakları şahin ton içeriyordu, küresel gerilim arttı.",
        "Fed yetkilileri enflasyonun kontrol altına alındığını belirtti, rahatlatıcı mesaj.",
        "Dolar endeksi güçleniyor, gelişen piyasa paraları baskı altında.",
    ],
    "yerel": [
        "Cari açık verileri beklentinin üzerinde geldi, TL üzerinde baskı artıyor.",
        "Sanayi üretimi yükseldi, büyüme rakamları olumlu, ekonomi yolunda.",
        "Jeopolitik gerilimler sınırda tırmandı, savunma harcamaları tartışılıyor.",
        "Kredi derecelendirme kuruluşu Türkiye notunu teyit etti, görünüm stabil.",
        "Seçim belirsizliği piyasaları olumsuz etkiliyor, yatırımcı temkinli.",
        "İhracat rekor kırdı, dış ticaret dengesi iyileşiyor.",
        "Bütçe açığı hedefin üzerine çıktı, mali disiplin sorgulanıyor.",
    ]
}

# Ağırlıklı keyword sözlüğü: kelime -> belirsizlik etkisi
BELIRSIZLIK_SOZLUGU = {
    # Yüksek belirsizlik (0.7 - 1.0)
    "tedirgin": 0.85, "gerilim": 0.90, "belirsizlik": 0.80,
    "kriz": 0.95, "çatışma": 0.92, "baskı": 0.75,
    "risk": 0.78, "eridi": 0.88, "düşüş": 0.72,
    "şahin": 0.70, "temkinli": 0.68, "endişe": 0.82,
    "çıkış": 0.74, "tırmandı": 0.88, "olumsuz": 0.76,
    "sorgulanıyor": 0.71, "zayıfladı": 0.73,
    # Düşük belirsizlik (0.0 - 0.3)
    "stabil": 0.15, "yolunda": 0.12, "olumlu": 0.18,
    "güçlü": 0.20, "iyileşiyor": 0.10, "rekor": 0.14,
    "rahatlatıcı": 0.08, "büyüme": 0.22, "teyit": 0.16,
    "kontrol": 0.19,
}

def haber_uret(n: int, kaynak_dagilim: dict = None) -> list[dict]:
    """
    Mock haber üretici. İleride bu fonksiyon GDELT API veya
    gerçek FED/TCMB PDF raporlarını parse eden bir modülle değiştirilecek.
    """
    if kaynak_dagilim is None:
        kaynak_dagilim = {"tcmb": 0.4, "fed": 0.3, "yerel": 0.3}

    haberler = []
    kaynaklar = list(kaynak_dagilim.keys())
    olasiliklar = list(kaynak_dagilim.values())

    for _ in range(n):
        kaynak = np.random.choice(kaynaklar, p=olasiliklar)
        metin = np.random.choice(MOCK_HABERLER[kaynak])
        haberler.append({"kaynak": kaynak, "metin": metin})
    return haberler


def belirsizlik_skoru_hesapla(metin: str) -> float:
    """
    Ağırlıklı keyword bazlı belirsizlik skoru.
    İleride bu fonksiyon LLM (Llama 3 vb.) çıktısıyla değiştirilecek.

    Placeholder API şablonu:
        def llm_belirsizlik_skoru(metin: str, model="llama3") -> float:
            prompt = f"Aşağıdaki haberin politik belirsizlik skorunu 0-1 arasında ver:\\n{metin}"
            response = llm_client.generate(prompt, model=model)
            return float(response.strip())
    """
    metin_lower = metin.lower()
    bulunan_skorlar = []

    for kelime, skor in BELIRSIZLIK_SOZLUGU.items():
        if kelime in metin_lower:
            bulunan_skorlar.append(skor)

    if not bulunan_skorlar:
        return np.random.uniform(0.3, 0.6)

    base = np.mean(bulunan_skorlar)
    noise = np.random.uniform(-0.05, 0.05)
    return np.clip(base + noise, 0.0, 1.0)


def nlp_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Haberleri üretir, skorlar ve DataFrame'e ekler."""
    haberler = haber_uret(len(df))

    df['News_Source'] = [h['kaynak'] for h in haberler]
    df['News_Text'] = [h['metin'] for h in haberler]
    df['Uncertainty_Score'] = df['News_Text'].apply(belirsizlik_skoru_hesapla)

    window = CONFIG["rolling_window"]
    df['Uncertainty_SMA'] = df['Uncertainty_Score'].rolling(window=window, min_periods=1).mean()

    return df

# ══════════════════════════════════════════════════════════════════════════════
# FAZ 3: GARCH MODELLEMESİ + KARŞILAŞTIRMA
# ══════════════════════════════════════════════════════════════════════════════

def garch_modelle(df: pd.DataFrame) -> dict:
    """
    Hem standart GARCH(1,1) hem GARCH-X modelini kurar.
    AIC/BIC/LogLik karşılaştırması yapar.
    """
    y = df['Return']
    x = df[['Uncertainty_Score']]

    # Standart GARCH(1,1) - baseline
    am_base = arch_model(y, mean='AR', vol='Garch',
                         p=CONFIG["garch_p"], q=CONFIG["garch_q"], rescale=False)
    res_base = am_base.fit(disp="off")

    # GARCH-X: Belirsizlik skoru dışsal değişken olarak
    am_x = arch_model(y, x=x, mean='ARX', vol='Garch',
                      p=CONFIG["garch_p"], q=CONFIG["garch_q"], rescale=False)
    res_x = am_x.fit(disp="off")

    df['Vol_GARCH'] = res_base.conditional_volatility
    df['Vol_GARCH_X'] = res_x.conditional_volatility

    window = CONFIG["rolling_window"]
    df['Realized_Vol'] = df['Return'].rolling(window=window).std()

    karsilastirma = pd.DataFrame({
        'Model': ['GARCH(1,1)', 'GARCH-X(1,1)'],
        'AIC': [res_base.aic, res_x.aic],
        'BIC': [res_base.bic, res_x.bic],
        'Log-Likelihood': [res_base.loglikelihood, res_x.loglikelihood],
        'Parametre Sayısı': [res_base.num_params, res_x.num_params],
    })

    return {
        "res_base": res_base,
        "res_x": res_x,
        "karsilastirma": karsilastirma,
        "df": df,
    }

# ══════════════════════════════════════════════════════════════════════════════
# FAZ 4: GELİŞMİŞ GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════════════════

def gorsellestir(df: pd.DataFrame, karsilastirma: pd.DataFrame):
    """6 panelli gelişmiş görselleştirme."""
    plt.rcParams.update({
        'figure.facecolor': '#f8f9fa',
        'axes.facecolor': '#ffffff',
        'axes.edgecolor': '#dee2e6',
        'grid.color': '#e9ecef',
        'font.size': 9,
    })

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

    # 1) Kapanış fiyatı
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df['Close'], color='#1a73e8', linewidth=1.5)
    ax1.set_title('BIST 100 Kapanış Fiyatı', fontweight='bold')
    ax1.set_ylabel('Fiyat (TL)')
    ax1.grid(True, alpha=0.4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

    # 2) Günlük getiriler
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#e53935' if r < 0 else '#43a047' for r in df['Return']]
    ax2.bar(df.index, df['Return'], color=colors, alpha=0.7, width=1.5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('Günlük Getiriler (%)', fontweight='bold')
    ax2.set_ylabel('Getiri (%)')
    ax2.grid(True, alpha=0.4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

    # 3) Belirsizlik skoru + SMA
    ax3 = fig.add_subplot(gs[1, :])
    kaynak_renk = {'tcmb': '#e53935', 'fed': '#1a73e8', 'yerel': '#ff9800'}
    for kaynak in kaynak_renk:
        mask = df['News_Source'] == kaynak
        if mask.any():
            ax3.scatter(df.index[mask], df['Uncertainty_Score'][mask],
                        color=kaynak_renk[kaynak], label=kaynak.upper(),
                        alpha=0.6, s=30, zorder=3)
    ax3.plot(df.index, df['Uncertainty_SMA'], color='black',
             linewidth=2, label=f'SMA({CONFIG["rolling_window"]})', zorder=4)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('NLP Politik Belirsizlik Skoru (Kaynak Bazlı)', fontweight='bold')
    ax3.set_ylabel('Skor (0-1)')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.4)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))

    # 4) Volatilite karşılaştırması: GARCH vs GARCH-X vs Realized
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(df.index, df['Vol_GARCH'], color='#7b1fa2', linewidth=1.5,
             label='GARCH(1,1)', alpha=0.8)
    ax4.plot(df.index, df['Vol_GARCH_X'], color='#e53935', linewidth=2,
             label='GARCH-X(1,1)', zorder=5)
    ax4.plot(df.index, df['Realized_Vol'], color='#90a4ae', linewidth=1,
             linestyle='--', label=f'Realized Vol ({CONFIG["rolling_window"]}gün)', alpha=0.7)
    ax4.set_title('Volatilite Karşılaştırması', fontweight='bold')
    ax4.set_ylabel('Volatilite')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.4)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))

    # 5) Belirsizlik vs Volatilite scatter
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.scatter(df['Uncertainty_Score'], df['Vol_GARCH_X'],
                alpha=0.5, s=25, color='#e53935', edgecolors='white', linewidth=0.3)
    z = np.polyfit(df['Uncertainty_Score'], df['Vol_GARCH_X'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Uncertainty_Score'].min(), df['Uncertainty_Score'].max(), 100)
    ax5.plot(x_line, p(x_line), color='black', linewidth=1.5, linestyle='--')
    corr = df['Uncertainty_Score'].corr(df['Vol_GARCH_X'])
    ax5.set_title(f'Belirsizlik vs Volatilite (r={corr:.3f})', fontweight='bold')
    ax5.set_xlabel('Uncertainty Score')
    ax5.set_ylabel('GARCH-X Volatilite')
    ax5.grid(True, alpha=0.4)

    # 6) Model karşılaştırma tablosu
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    tablo_veri = karsilastirma.round(2).values.tolist()
    tablo = ax6.table(
        cellText=tablo_veri,
        colLabels=karsilastirma.columns.tolist(),
        cellLoc='center',
        loc='center',
    )
    tablo.auto_set_font_size(False)
    tablo.set_fontsize(9)
    tablo.scale(1.0, 1.8)
    for (row, col), cell in tablo.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1a73e8')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else '#ffffff')
    ax6.set_title('Model Karşılaştırması (AIC/BIC)', fontweight='bold', pad=20)

    fig.suptitle('LLM Tabanlı Politika Olay Çıkarımı ve Finansal Volatilite Modellemesi',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(CONFIG["output_plot"], dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Grafikler kaydedildi: {CONFIG['output_plot']}")

# ══════════════════════════════════════════════════════════════════════════════
# FAZ 5: EXPORT + DİAGNOSTİCS
# ══════════════════════════════════════════════════════════════════════════════

def diagnostics_yazdir(sonuclar: dict):
    """Model diagnostics ve karşılaştırma tablosunu yazdırır."""
    print("\n" + "=" * 70)
    print("MODEL KARŞILAŞTIRMASI")
    print("=" * 70)
    print(sonuclar["karsilastirma"].to_string(index=False))

    print("\n" + "=" * 70)
    print("GARCH-X MODEL ÖZETİ")
    print("=" * 70)
    print(sonuclar["res_x"].summary())

    df = sonuclar["df"]
    corr = df['Uncertainty_Score'].corr(df['Vol_GARCH_X'])
    print(f"\n  Belirsizlik-Volatilite Korelasyonu: {corr:.4f}")

    # Uncertainty Score'un p-value kontrolü
    params = sonuclar["res_x"].pvalues
    if 'Uncertainty_Score' in params.index:
        pval = params['Uncertainty_Score']
        anlamli = "EVET ✓" if pval < 0.05 else "HAYIR ✗"
        print(f"  Uncertainty_Score p-value: {pval:.4f} → İstatistiksel anlamlılık: {anlamli}")

    aic_base = sonuclar["res_base"].aic
    aic_x = sonuclar["res_x"].aic
    kazanan = "GARCH-X" if aic_x < aic_base else "GARCH"
    print(f"  AIC Karşılaştırması: GARCH={aic_base:.2f} vs GARCH-X={aic_x:.2f} → Kazanan: {kazanan}")


def export_csv(df: pd.DataFrame):
    """Sonuçları CSV'ye kaydeder."""
    export_cols = ['Close', 'Return', 'News_Source', 'News_Text',
                   'Uncertainty_Score', 'Uncertainty_SMA',
                   'Vol_GARCH', 'Vol_GARCH_X', 'Realized_Vol']
    df[export_cols].to_csv(CONFIG["output_csv"], encoding='utf-8-sig')
    print(f"  Veriler kaydedildi: {CONFIG['output_csv']}")

# ══════════════════════════════════════════════════════════════════════════════
# ANA ÇALIŞTIRMA
# ══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(CONFIG["seed"])

    print("=" * 70)
    print("  BİTİRME PROJESİ DEMO - v2.0")
    print("  LLM Tabanlı Politika Olay Çıkarımı ve Finansal Volatilite")
    print("=" * 70)

    print("\n[1/5] Finansal veriler çekiliyor...")
    df = veri_cek(CONFIG["ticker"], CONFIG["period"], CONFIG["interval"])
    print(f"  {len(df)} günlük veri çekildi ({df.index[0].strftime('%d.%m.%Y')} - {df.index[-1].strftime('%d.%m.%Y')})")

    print("[2/5] Politik olay çıkarımı (Mock NLP)...")
    df = nlp_pipeline(df)
    print(f"  Ortalama Belirsizlik Skoru: {df['Uncertainty_Score'].mean():.3f}")
    for src in df['News_Source'].unique():
        cnt = (df['News_Source'] == src).sum()
        avg = df.loc[df['News_Source'] == src, 'Uncertainty_Score'].mean()
        print(f"    {src.upper():>6}: {cnt} haber, ort. skor={avg:.3f}")

    print("[3/5] GARCH modelleri eğitiliyor...")
    sonuclar = garch_modelle(df)
    df = sonuclar["df"]

    print("[4/5] Görselleştirme oluşturuluyor...")
    gorsellestir(df, sonuclar["karsilastirma"])

    print("[5/5] Sonuçlar kaydediliyor...")
    export_csv(df)
    diagnostics_yazdir(sonuclar)

    print("\n" + "=" * 70)
    print("  DEMO TAMAMLANDI")
    print("=" * 70)


if __name__ == "__main__":
    main()
