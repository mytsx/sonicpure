# Kurulum Rehberi

## Diğer Projelerde Kullanım

### Seçenek 1: GitHub Private Repo (Önerilen)

```bash
# requirements.txt'e ekle:
git+ssh://git@github.com/mytsx/sonicpure.git

# Veya belirli bir versiyon:
git+ssh://git@github.com/mytsx/sonicpure.git@v0.1.0
```

**Python kodunda:**
```python
from sonicpure import AudioPipeline

pipeline = AudioPipeline()
result = pipeline.process("input.wav", "output.wav")
```

### Seçenek 2: Local Development

Aynı makinede birden fazla proje geliştiriyorsanız:

```bash
# Diğer projenin requirements.txt:
-e /Users/yerli/Developer/tools/sonicpure

# Veya direkt kurulum:
cd /path/to/other/project
pip install -e /Users/yerli/Developer/tools/sonicpure
```

### Seçenek 3: PyPI (Public)

```bash
# PyPI'ye publish ettikten sonra:
pip install sonicpure
```

## GitHub'a Push

```bash
# 1. GitHub'da public repo oluştur
# 2. Remote ekle ve push et
git remote add origin git@github.com:mytsx/sonicpure.git
git push -u origin main

# Tag oluştur (versiyonlama)
git tag -a v0.1.0 -m "Initial release"
git push origin v0.1.0
```

## requirements.txt Örnekleri

### GitHub Public Repo
```txt
# requirements.txt
numpy>=1.20.0
scipy>=1.7.0
git+ssh://git@github.com/mytsx/sonicpure.git@v0.1.0
```

### Local Development
```txt
# requirements.txt
-e /Users/yerli/Developer/tools/sonicpure
```

### PyPI
```txt
# requirements.txt
sonicpure==0.1.0
```

## Kullanım Örneği

```python
# my_other_project/main.py
from sonicpure import AudioPipeline

def process_audio(input_file):
    pipeline = AudioPipeline(
        engine='speechbrain',
        max_silence=0.5
    )

    output_file = input_file.replace('.wav', '_clean.wav')
    result = pipeline.process(input_file, output_file)

    print(f"✅ Processed: {result['time_saved']:.2f}s saved")
    return output_file

# Kullanım
clean_audio = process_audio("recording.wav")
```

## Troubleshooting

### Import hatası alıyorsanız:
```bash
# Paketin kurulu olduğunu kontrol edin
pip list | grep sonicpure

# Yoksa tekrar kurun
pip install -e /path/to/sonicpure
```

### GitHub private repo authentication:
```bash
# SSH key kullanın (önerilen)
git+ssh://git@github.com/username/repo.git

# HTTPS + token:
git+https://TOKEN@github.com/username/repo.git
```
