# Gürültü Temizleme Test Sonuçları

Test Dosyası: `tts_fbea8465-85d5-44cf-9f6d-779a1e7c31c2.wav`
- Süre: 49.48 saniye
- Sample Rate: 24000 Hz
- Kanal: Mono (1)
- Boyut: 4.5 MB

---

## Test Edilen Motorlar

### 1. noisereduce (En Kolay)

**Kurulum:**
```bash
pip install noisereduce soundfile numpy
```

**Kullanım:**
```bash
python3 test_noisereduce.py
```

**Sonuçlar:**
- ✅ **Stationary Mode** (Agresif)
  - Çıktı: `output_tests/noisereduce_stationary.wav` (2.3 MB)
  - İşlem Süresi: **0.16 saniye** ⚡️
  - Özellik: Stationary (sabit) gürültü için daha agresif temizlik

- ✅ **Non-Stationary Mode** (Yumuşak)
  - Çıktı: `output_tests/noisereduce_nonstationary.wav` (2.3 MB)
  - İşlem Süresi: **0.14 saniye** ⚡️
  - Özellik: Değişken gürültü için daha yumuşak temizlik

**Artılar:**
- Çok hızlı
- Kurulumu çok kolay
- Python-only, external dependency yok
- İki farklı mod sunuyor

**Eksiler:**
- Basit spektral subtraction yöntemi
- Çok agresif ayarlarda ses kalitesi düşebilir

---

### 2. RNNoise (Orta Seviye)

**Kurulum:**
```bash
pip install pyrnnoise scipy
```

**Kullanım:**
```bash
python3 test_rnnoise.py
```

**Sonuçlar:**
- ✅ Çıktı: `output_tests/rnnoise_cleaned.wav` (2.3 MB)
- İşlem Süresi: **2.78 saniye** 🚀
- Frame'ler: 4000+ frame işlendi

**Artılar:**
- RNN (Recurrent Neural Network) tabanlı
- Konuşma için özel optimize edilmiş
- Gerçek zamanlı kullanım için tasarlanmış
- Mozilla tarafından geliştirilmiş, iyi test edilmiş

**Eksiler:**
- noisereduce'dan biraz daha yavaş
- Frame bazlı işlem (daha fazla kod)

---

### 3. SpeechBrain (En Güçlü)

**Kurulum:**
```bash
pip install speechbrain
```

**Kullanım:**
```bash
python3 test_speechbrain.py
```

**Sonuçlar:**
- ✅ Çıktı: `output_tests/speechbrain_enhanced.wav` (1.5 MB)
- İşlem Süresi: **6.23 saniye** 🐢
- Model: MetricGAN+ (pre-trained)

**Artılar:**
- En gelişmiş deep learning modeli
- MetricGAN+ state-of-the-art enhancement
- Teorik olarak en iyi kalite
- Akademik araştırmalarda kullanılıyor

**Eksiler:**
- En yavaş seçenek
- İlk çalıştırmada model indirir (~100-200 MB)
- GPU olmadan yavaş olabilir
- En fazla bağımlılık gerektiriyor

---

## Performans Karşılaştırması

| Motor | İşlem Süresi | Hız | Dosya Boyutu | Kurulum Kolaylığı |
|-------|--------------|-----|--------------|-------------------|
| **noisereduce** (stationary) | 0.16s | ⚡️⚡️⚡️⚡️⚡️ | 2.3 MB | ⭐️⭐️⭐️⭐️⭐️ |
| **noisereduce** (non-stat.) | 0.14s | ⚡️⚡️⚡️⚡️⚡️ | 2.3 MB | ⭐️⭐️⭐️⭐️⭐️ |
| **RNNoise** | 2.78s | ⚡️⚡️⚡️⚡️ | 2.3 MB | ⭐️⭐️⭐️⭐️ |
| **SpeechBrain** | 6.23s | ⚡️⚡️ | 1.5 MB | ⭐️⭐️⭐️ |

---

## Öneriler

### Batch İşleme İçin:
- **noisereduce**: Yüzlerce dosya için en hızlı
- **RNNoise**: Hız-kalite dengesi için

### Kalite Öncelikli:
- **SpeechBrain**: En iyi kalite (ama yavaş)
- **RNNoise**: İyi kalite ve makul hız

### Gerçek Zamanlı:
- **RNNoise**: Gerçek zamanlı işlem için tasarlanmış
- **noisereduce**: Çok hızlı ama kalite düşük olabilir

---

## Sonraki Adımlar

1. **Dinleme Testi**: Tüm çıktı dosyalarını dinleyip karşılaştırın
2. **Favori Seçimi**: Hangi motorun sesini daha iyi bulduğunuzu belirleyin
3. **Entegrasyon**: Tüm motorları tek bir CLI aracında birleştirin
4. **Sessizlik İşleme**: PRD'deki sessizlik kısaltma özelliğini ekleyin

---

## Çıktı Dosyaları

Tüm test çıktıları `output_tests/` klasöründe:

```
output_tests/
├── noisereduce_stationary.wav      (2.3 MB) - Agresif
├── noisereduce_nonstationary.wav   (2.3 MB) - Yumuşak
├── rnnoise_cleaned.wav             (2.3 MB) - RNN tabanlı
└── speechbrain_enhanced.wav        (1.5 MB) - Deep learning
```

**Not:** SpeechBrain'in dosya boyutu daha küçük çünkü farklı encoding kullanıyor olabilir.
