# LLM Tabanlı Politika Olay Çıkarımı ve Finansal Volatilite Modellemesi

Bitirme projesi — Haber metinlerinden LLM ile **politik/ekonomik belirsizlik skoru** çıkarıp bunu **GARCH-X ailesi** volatilite modellerinde dışsal değişken olarak kullanan uçtan uca pipeline.

```
Haber (TCMB/FED/GDELT) → LLM (Llama 3) → Belirsizlik Skoru (0-1)
                                                ↓
Finansal veri (BIST 100)  →  GARCH/EGARCH/TARCH  +  -X varyantları
                                                ↓
                         Dashboard + XAI (variance decomposition)
```

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Opsiyonel: Gerçek LLM için Ollama
brew install ollama
ollama pull llama3
```

## Hızlı Başlangıç

```bash
# Tam pipeline (gerçek TCMB+FED+GDELT scraper)
python main.py

# Hızlı test (mevcut DB verisi + mock NLP)
python main.py --skip-fetch --force-mock

# Farklı ticker
python main.py --ticker "^GSPC" --mock-only --force-mock

# Dashboard'u aç
open outputs/figures/dashboard.html
```

### CLI argümanları

| Flag | Açıklama |
|---|---|
| `--ticker SYMBOL` | yfinance ticker (varsayılan: XU100.IS) |
| `--skip-fetch` | Veri çekmeyi atla (DB'deki mevcut veriyi kullan) |
| `--skip-nlp` | NLP adımını atla |
| `--skip-viz` | Görselleştirmeyi atla |
| `--skip-oos` | Out-of-sample değerlendirmeyi atla |
| `--force-mock` | Ollama yerine MockNLP kullan |
| `--mock-only` | Gerçek haber scraper'larını atla |
| `--forecast N` | N günlük tahmin ufku (varsayılan: 5) |
| `--test-ratio F` | OOS test oranı (varsayılan: 0.20) |

## Klasör Yapısı

```
cursorProject/
├── config.yaml              # Tüm parametreler
├── main.py                  # Pipeline orkestratör
├── requirements.txt
├── src/
│   ├── pipeline/            # DB + veri çekme + önişleme
│   ├── nlp/                 # LLM + prompts + MockNLP + EventExtractor
│   ├── models/              # GARCH/EGARCH/TARCH + benchmark + OOS
│   └── visualization/       # Plotly dashboard + XAI
├── tests/                   # 4 modül entegrasyon testi
├── data/                    # SQLite DB (git'te yok, pipeline üretir)
└── outputs/figures/         # Dashboard + XAI grafikleri
```

## Modüller

- **Pipeline**: SQLite DB, yfinance, TCMB/FED scraper, GDELT API, mock fallback, preprocessing.
- **NLP**: Ollama (Llama 3) REST wrapper, 13 olay türlü JSON prompt, rule-based MockNLP fallback.
- **Models**: 6 model (GARCH, EGARCH, TARCH + -X varyantları), AIC/BIC sıralama, Ljung-Box + Jarque-Bera, expanding-window OOS (RMSE/MAE/QLIKE/HitRate).
- **Visualization**: 6 panelli Plotly dashboard + variance decomposition + sensitivity + event timeline.

## Test

```bash
python tests/test_module1.py   # Veri pipeline
python tests/test_module2.py   # NLP olay çıkarımı
python tests/test_module3.py   # GARCH modelleme
python tests/test_module4.py   # Dashboard + XAI
```

## Durum

Detaylı durum ve yapılacaklar için bkz. [`PROJE_DURUMU.txt`](PROJE_DURUMU.txt).
