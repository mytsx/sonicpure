# SonicPure 🎙️

**Audio Noise Reduction & Enhancement Library**

Profesyonel ses temizleme kütüphanesi - çoklu motor desteği, otomatik sessizlik kısaltma ve akıllı normalizasyon.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Özellikler

- 🎯 **3 Güçlü Gürültü Giderme Motoru**
  - SpeechBrain MetricGAN+ (en kaliteli)
  - RNNoise (hızlı, gerçek zamanlı)
  - NoiseReduce (basit, hızlı)

- 🔇 **Akıllı Sessizlik Kısaltma**
  - Otomatik sessizlik tespiti (RMS-based)
  - Uzun sessizlikleri 0.5 saniyeye kısaltır
  - %2-13 arası süre tasarrufu

- 📊 **Gentle Normalization**
  - Referans dosya bazlı RMS normalizasyonu
  - Tanh-based yumuşak limiting (analog tarzı)
  - Pik sesleri önler, doğal kaliteyi korur

- ⚡ **Hızlı & Verimli**
  - 50 saniyelik ses → 2 saniyede işlenir
  - Otomatik sample rate dönüşümü
  - Geçici dosyalar otomatik temizlenir

## 📦 Kurulum

### Pip ile (Önerilen)

```bash
# Development modunda kurulum (düzenlenebilir)
pip install -e .

# Normal kurulum
pip install .
```

### Manuel Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Kütüphaneyi Python path'ine ekle
export PYTHONPATH="${PYTHONPATH}:/path/to/sonicpure"
```

## 🚀 Hızlı Başlangıç

### CLI Kullanımı

```bash
# Basit kullanım
python clean_audio.py input.wav output.wav

# Farklı motor seç
python clean_audio.py input.wav output.wav --engine rnnoise

# Daha agresif sessizlik kısaltma
python clean_audio.py input.wav output.wav --max-silence 0.3

# Ara dosyaları kaydet (debug için)
python clean_audio.py input.wav output.wav --save-intermediate

# Normalizasyon olmadan
python clean_audio.py input.wav output.wav --no-normalize
```

### Python Kütüphanesi Olarak

```python
from sonicpure import AudioPipeline

# Basit kullanım
pipeline = AudioPipeline()
result = pipeline.process("input.wav", "output.wav")

print(f"Original: {result['original_duration']:.2f}s")
print(f"Final: {result['final_duration']:.2f}s")
print(f"Saved: {result['time_saved']:.2f}s")
```

### İleri Seviye Örnekler

```python
from sonicpure import AudioPipeline

# Özelleştirilmiş pipeline
pipeline = AudioPipeline(
    engine='speechbrain',        # veya 'rnnoise', 'noisereduce'
    max_silence=0.5,             # maksimum sessizlik süresi
    silence_threshold_db=-40,    # sessizlik eşiği
    normalize=True,              # normalizasyon uygula
    reference_file='ref.wav'     # referans dosya (opsiyonel)
)

result = pipeline.process("input.wav", "output.wav")

# Detaylı sonuçlar
print(f"Engine: {result['engine']}")
print(f"Processing time: {result['processing_time']:.2f}s")
print(f"Time saved: {result['time_saved']:.2f}s ({result['time_saved']/result['original_duration']*100:.1f}%)")
```

### Motorları Ayrı Ayrı Kullanma

```python
from sonicpure.engines import SpeechBrainEngine, RNNoiseEngine, NoiseReduceEngine
from sonicpure.processors import SilenceTrimmer, AudioNormalizer

# Sadece gürültü temizleme
engine = SpeechBrainEngine()
engine.process("input.wav", "denoised.wav")

# Sadece sessizlik kısaltma
trimmer = SilenceTrimmer()
trimmer.trim("input.wav", "trimmed.wav", max_silence_duration=0.5)

# Sadece normalizasyon
normalizer = AudioNormalizer()
normalizer.normalize("input.wav", "normalized.wav", reference_file="ref.wav")
```

## 🎛️ Motorlar Karşılaştırması

| Motor | Hız | Kalite | Kullanım Durumu |
|-------|-----|--------|-----------------|
| **SpeechBrain** | 🐢 Yavaş | ⭐️⭐️⭐️⭐️⭐️ En İyi | Podcast, video dubbing, profesyonel kullanım |
| **RNNoise** | 🚀 Hızlı | ⭐️⭐️⭐️⭐️ Çok İyi | Gerçek zamanlı, streaming, batch işleme |
| **NoiseReduce** | ⚡️ Çok Hızlı | ⭐️⭐️⭐️ İyi | Hızlı önizleme, basit temizlik |

## 📊 Test Sonuçları

Gerçek TTS ses dosyaları üzerinde test edildi:

| Dosya | Süre | İşlem | Tasarruf | Sonuç |
|-------|------|-------|----------|-------|
| test0.wav | 3.32s | 1.40s | 0.42s (%12.7) | ✅ Çok iyi |
| test1.wav | 43.88s | 2.26s | 1.25s (%2.8) | ✅ Mükemmel |
| tts_original.wav | 49.48s | 8.70s | 2.04s (%4.1) | ✅ Harika |

## 🛠️ Teknik Detaylar

### Pipeline Adımları

1. **Noise Reduction**
   - SpeechBrain: 16kHz model, otomatik resampling
   - RNNoise: Frame-based processing, Savitzky-Golay smoothing
   - NoiseReduce: Spectral subtraction, prop_decrease=0.7

2. **Silence Trimming**
   - RMS energy-based detection
   - -40 dB threshold (ayarlanabilir)
   - Minimum 0.3s sessizlik tespiti
   - Maksimum 0.5s sessizlik bırakır

3. **Normalization**
   - Referans dosya bazlı RMS matching
   - Safe gain hesaplama (clipping önleme)
   - Tanh-based gentle limiter (threshold: 0.95)
   - Max peak: -0.5 dBFS (güvenlik için)

### Ses Kalitesi Garantisi

- ✅ Pik sesler engellenir (gentle limiting)
- ✅ Doğal ses karakteri korunur
- ✅ Artifact-free işlem
- ✅ Clipping önleme garantili

## 📁 Proje Yapısı

```
sonicpure/
├── sonicpure/               # Ana kütüphane
│   ├── __init__.py         # Paket tanımı
│   ├── engines.py          # Gürültü temizleme motorları
│   ├── processors.py       # Trimmer & Normalizer
│   └── pipeline.py         # All-in-one pipeline
├── clean_audio.py          # CLI aracı
├── test_*.py               # Test scriptleri
├── audio_analyzer.py       # Ses analiz aracı
├── audio_normalizer.py     # Standalone normalizer
├── silence_trimmer.py      # Standalone trimmer
├── setup.py                # Paket kurulum
├── requirements.txt        # Bağımlılıklar
├── README.md              # Bu dosya
└── prd.md                 # Proje gereksinimleri
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [SpeechBrain](https://speechbrain.github.io/) - MetricGAN+ modeli
- [RNNoise](https://github.com/xiph/rnnoise) - RNN-based denoising
- [noisereduce](https://github.com/timsainb/noisereduce) - Spectral gating

## 📧 İletişim

Mehmet Yerli - iletisim@mehmetyerli.com.tr

Proje Linki: [https://github.com/mehmetyerli/sonicpure](https://github.com/mehmetyerli/sonicpure)

---

⭐️ Projeyi beğendiyseniz yıldız vermeyi unutmayın!
