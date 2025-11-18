#!/usr/bin/env python3
"""
Audio Analyzer - Ses dosyalarının seviye ve kalite analizini yapar
"""

import numpy as np
import soundfile as sf
from pathlib import Path

def analyze_audio(file_path: str) -> dict:
    """
    Ses dosyasını analiz et

    Returns:
        dict: Analiz sonuçları (max_db, rms_db, peak, duration vb.)
    """
    print(f"\n[Analyzer] Analyzing: {file_path}")

    # Ses dosyasını yükle
    audio, rate = sf.read(file_path)

    # Duration
    duration = len(audio) / rate

    # Peak (maksimum değer)
    peak = np.max(np.abs(audio))

    # RMS (Root Mean Square) - ortalama enerji
    rms = np.sqrt(np.mean(audio**2))

    # dB'ye çevir (referans: 1.0 = 0 dBFS)
    # dBFS = 20 * log10(level / reference)
    # reference = 1.0 (full scale)

    if peak > 0:
        peak_db = 20 * np.log10(peak)
    else:
        peak_db = -np.inf

    if rms > 0:
        rms_db = 20 * np.log10(rms)
    else:
        rms_db = -np.inf

    # Crest factor (peak to RMS ratio)
    if rms > 0:
        crest_factor = peak / rms
        crest_factor_db = 20 * np.log10(crest_factor)
    else:
        crest_factor = np.inf
        crest_factor_db = np.inf

    # Dynamic range (rough estimate)
    # En sessiz ve en yüksek bölgelerin farkı
    frame_size = int(0.1 * rate)  # 100ms frames
    frame_rms = []

    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i+frame_size]
        frame_rms.append(np.sqrt(np.mean(frame**2)))

    frame_rms = np.array(frame_rms)
    frame_rms = frame_rms[frame_rms > 0]  # Sıfırları kaldır

    if len(frame_rms) > 0:
        min_rms = np.min(frame_rms)
        max_rms = np.max(frame_rms)
        dynamic_range = 20 * np.log10(max_rms / min_rms) if min_rms > 0 else 0
    else:
        dynamic_range = 0

    result = {
        'file': file_path,
        'duration': duration,
        'sample_rate': rate,
        'peak': peak,
        'peak_db': peak_db,
        'rms': rms,
        'rms_db': rms_db,
        'crest_factor': crest_factor,
        'crest_factor_db': crest_factor_db,
        'dynamic_range_db': dynamic_range
    }

    # Sonuçları yazdır
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample Rate: {rate} Hz")
    print(f"  Peak Level: {peak:.4f} ({peak_db:.2f} dBFS)")
    print(f"  RMS Level: {rms:.4f} ({rms_db:.2f} dBFS)")
    print(f"  Crest Factor: {crest_factor:.2f} ({crest_factor_db:.2f} dB)")
    print(f"  Dynamic Range: {dynamic_range:.2f} dB")

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("AUDIO LEVEL ANALYZER")
    print("=" * 70)

    # Analiz edilecek dosyalar
    files = [
        "tts_fbea8465-85d5-44cf-9f6d-779a1e7c31c2.wav",  # Orijinal
        "output_tests/speechbrain_enhanced.wav",          # SpeechBrain
        "output_tests/speechbrain_trimmed_0.5s.wav",      # Trimmed
    ]

    results = []
    for file_path in files:
        if Path(file_path).exists():
            result = analyze_audio(file_path)
            results.append(result)
        else:
            print(f"\n[Analyzer] File not found: {file_path}")

    # Karşılaştırmalı tablo
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'File':<45} {'Peak dBFS':>12} {'RMS dBFS':>12}")
    print("-" * 70)

    for result in results:
        file_name = Path(result['file']).name
        print(f"{file_name:<45} {result['peak_db']:>12.2f} {result['rms_db']:>12.2f}")

    # Fark analizi
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("DIFFERENCE ANALYSIS")
        print("=" * 70)

        original = results[0]

        for i, result in enumerate(results[1:], 1):
            file_name = Path(result['file']).name
            peak_diff = result['peak_db'] - original['peak_db']
            rms_diff = result['rms_db'] - original['rms_db']

            print(f"\n{file_name} vs Original:")
            print(f"  Peak difference: {peak_diff:+.2f} dB")
            print(f"  RMS difference: {rms_diff:+.2f} dB")

            if peak_diff < -3:
                print(f"  ⚠️  Peak is {abs(peak_diff):.1f} dB lower (might need normalization)")
            elif peak_diff > -1:
                print(f"  ✓ Peak level is good")
