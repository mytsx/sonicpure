# Gürültü Temizleme Uygulaması – PRD

## 1. Ürün Özeti

Bu proje, ses kayıtlarındaki gürültüleri mümkün olduğunca otomatik ve kaliteli biçimde temizleyen, **birden fazla açık kaynak gürültü giderme kütüphanesini** tek çatı altında toplayan bir Python uygulamasıdır.

**Amaç:**

- Farklı **noise reduction / speech enhancement** motorlarını (noisereduce, RNNoise, SpeechBrain vb.) aynı dosya üzerinde çalıştırmak,
- Ortaya çıkan sonuçları karşılaştırmak,
- Zamanla **kullanıcıya göre “favori motor” veya “en iyi preset”** belirleyebilmek,
- Batch (toplu) şekilde kullanılabilecek stabil bir araç sunmaktır.

Uygulama öncelikli olarak **CLI (komut satırı)** odaklı olacak, isterse üzerine GUI eklenebilir.

---

## 2. Hedef Kullanıcılar

- Eğitim/tanıtım videoları için ses kaydı alan içerik üreticileri
- Python bilen geliştiriciler (CI pipeline, otomasyon, batch işleme)
- Podcast, ders kaydı, ekran videosu vb. seslerini toplu temizlemek isteyenler

---

## 3. Hedefler

1. **Kaliteli gürültü temizliği**  
   En az 2–3 farklı motor kullanarak gürültüyü azaltma / konuşmayı iyileştirme.

2. **Karşılaştırma & Favori seçim**  
   Aynı input için farklı motorların çıktılarını üretmek, kullanıcıya dinletmek ve belirli bir motor/preset’i “favori” olarak işaretleyebilmek.

3. **Otomatik Sessizlik İşleme (özellikle isteniyor)**  
   - Sesin genel dağılımına göre **düşük seviyeli alanları** tespit et,
   - Bu alanları tamamen sessize çevir,
   - Sessiz alanlar 1 saniyeden uzunsa **1 saniyeye kısalt**.

4. **Batch işleme**  
   Bir klasör / liste içindeki yüzlerce ses dosyasını tek komutla işleyebilme.

5. **Modüler mimari**  
   Gürültü motorları “plugin” gibi olsun; yeni motor eklenebilsin.

---

## 4. Kapsam

### 4.1. Kapsam Dahilinde

- WAV başta olmak üzere yaygın formatlarda ses dosyası okuma/yazma (gerekirse ffmpeg ile).
- Aşağıdaki motorların desteklenmesi:
  - `noisereduce` (Python lib)
  - `RNNoise` (Python binding veya external binary ile)
  - `SpeechBrain` (hazır enhancement modeli)
  - (Opsiyonel) Basit filtreler: high-pass filter, normalizasyon
- Sessizlik tespiti ve kısaltma (librosa veya pydub ile).
- Komut satırı arayüzü.
- Basit YAML/JSON konfigürasyon.

### 4.2. Kapsam Dışı (ilk sürüm için)

- Gerçek zamanlı (live) mikrofon işleme GUI’si.
- Web arayüzü / server.
- Çok gelişmiş GUI (ileride eklenebilir).

---

## 5. Kullanım Senaryoları

### 5.1. Tek dosya temizleme

1. Kullanıcı:
   ```bash
   python clean_audio.py input.wav -o output_folder --preset aggressive