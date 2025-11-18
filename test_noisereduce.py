#!/usr/bin/env python3
"""
Test script for noisereduce library
En basit noise reduction motoru
"""

import numpy as np
import soundfile as sf
import noisereduce as nr
from pathlib import Path
import time

def clean_with_noisereduce(input_file: str, output_file: str, stationary: bool = True, prop_decrease: float = 0.7):
    """
    noisereduce kullanarak ses dosyasını temizle

    Args:
        input_file: Girdi WAV dosyası
        output_file: Çıktı WAV dosyası
        stationary: True ise stationary noise reduction (daha agresif)
        prop_decrease: Gürültü azaltma oranı (0.0-1.0)
                      0.5 = Yumuşak, 0.7 = Orta, 1.0 = Agresif
    """
    print(f"[noisereduce] Loading audio: {input_file}")

    # Ses dosyasını yükle
    data, rate = sf.read(input_file)
    print(f"[noisereduce] Sample rate: {rate} Hz")
    print(f"[noisereduce] Duration: {len(data)/rate:.2f} seconds")
    print(f"[noisereduce] Channels: {data.shape[1] if len(data.shape) > 1 else 1}")

    # Başlangıç zamanı
    start_time = time.time()

    # Noise reduction uygula
    print(f"[noisereduce] Applying noise reduction (stationary={stationary}, prop_decrease={prop_decrease})...")
    reduced_noise = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=stationary,
        prop_decrease=prop_decrease  # Gürültüyü ne kadar azaltacağız (0.0-1.0)
    )

    elapsed = time.time() - start_time
    print(f"[noisereduce] Processing completed in {elapsed:.2f} seconds")

    # Sonucu kaydet
    print(f"[noisereduce] Saving to: {output_file}")
    sf.write(output_file, reduced_noise, rate)

    return {
        'engine': 'noisereduce',
        'input_file': input_file,
        'output_file': output_file,
        'sample_rate': rate,
        'duration': len(data) / rate,
        'processing_time': elapsed,
        'stationary': stationary,
        'prop_decrease': prop_decrease
    }


if __name__ == "__main__":
    # Test dosyası
    input_wav = "tts_fbea8465-85d5-44cf-9f6d-779a1e7c31c2.wav"

    # Çıktı klasörü oluştur
    output_dir = Path("output_tests")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("NOISEREDUCE TEST - FARLI PROP_DECREASE DEĞERLERİ")
    print("=" * 60)

    results = []

    # 1. Yumuşak (0.5) - Sesi daha az kısar
    print("\n### Test 1: Gentle (prop_decrease=0.5)")
    result1 = clean_with_noisereduce(
        input_wav,
        str(output_dir / "noisereduce_gentle_0.5.wav"),
        stationary=True,
        prop_decrease=0.5
    )
    results.append(result1)

    # 2. Orta (0.7) - Dengeli
    print("\n### Test 2: Medium (prop_decrease=0.7)")
    result2 = clean_with_noisereduce(
        input_wav,
        str(output_dir / "noisereduce_medium_0.7.wav"),
        stationary=True,
        prop_decrease=0.7
    )
    results.append(result2)

    # 3. Agresif (1.0) - Maksimum temizlik
    print("\n### Test 3: Aggressive (prop_decrease=1.0)")
    result3 = clean_with_noisereduce(
        input_wav,
        str(output_dir / "noisereduce_aggressive_1.0.wav"),
        stationary=True,
        prop_decrease=1.0
    )
    results.append(result3)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. prop_decrease={result['prop_decrease']}:")
        print(f"   Output: {result['output_file']}")
        print(f"   Time: {result['processing_time']:.2f}s")

    print("\n" + "=" * 60)
    print("ÖNERİ:")
    print("=" * 60)
    print("• 0.5 = En yumuşak, sesi az kısar (daha doğal ama gürültü kalabilir)")
    print("• 0.7 = Orta seviye (dengeli)")
    print("• 1.0 = En agresif, sesi çok kısabilir")
    print("\n✓ Dosyaları dinleyip hangisi daha iyi belirleyin!")
    print(f"✓ Dosyalar: {output_dir}/ klasöründe")
