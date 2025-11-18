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

def clean_with_noisereduce(input_file: str, output_file: str, stationary: bool = True):
    """
    noisereduce kullanarak ses dosyasını temizle

    Args:
        input_file: Girdi WAV dosyası
        output_file: Çıktı WAV dosyası
        stationary: True ise stationary noise reduction (daha agresif)
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
    print(f"[noisereduce] Applying noise reduction (stationary={stationary})...")
    reduced_noise = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=stationary,
        prop_decrease=1.0  # Gürültüyü ne kadar azaltacağız (0.0-1.0)
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
        'stationary': stationary
    }


if __name__ == "__main__":
    # Test dosyası
    input_wav = "tts_fbea8465-85d5-44cf-9f6d-779a1e7c31c2.wav"

    # Çıktı klasörü oluştur
    output_dir = Path("output_tests")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("NOISEREDUCE TEST")
    print("=" * 60)

    # 1. Stationary mode (varsayılan - daha agresif)
    print("\n### Test 1: Stationary Mode (Aggressive)")
    result1 = clean_with_noisereduce(
        input_wav,
        str(output_dir / "noisereduce_stationary.wav"),
        stationary=True
    )

    # 2. Non-stationary mode (daha yumuşak)
    print("\n### Test 2: Non-Stationary Mode (Gentle)")
    result2 = clean_with_noisereduce(
        input_wav,
        str(output_dir / "noisereduce_nonstationary.wav"),
        stationary=False
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n1. Stationary mode:")
    print(f"   Output: {result1['output_file']}")
    print(f"   Time: {result1['processing_time']:.2f}s")

    print(f"\n2. Non-stationary mode:")
    print(f"   Output: {result2['output_file']}")
    print(f"   Time: {result2['processing_time']:.2f}s")

    print("\n✓ Her iki dosyayı dinleyip karşılaştırabilirsiniz!")
    print(f"✓ Dosyalar: {output_dir}/ klasöründe")
