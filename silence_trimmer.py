#!/usr/bin/env python3
"""
Silence Trimmer - Sessiz bölgeleri tespit edip kısaltır
SpeechBrain çıktısındaki temizlenmiş sesi alır, sessiz alanları 0.5 saniyeye kısaltır
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import time
from typing import List, Tuple

def detect_silence_regions(audio: np.ndarray, sample_rate: int,
                          threshold_db: float = -40,
                          min_silence_duration: float = 0.3) -> List[Tuple[int, int]]:
    """
    Sessiz bölgeleri tespit et

    Args:
        audio: Ses verisi (numpy array)
        sample_rate: Sample rate (Hz)
        threshold_db: Sessizlik eşiği (dB), -40 dB altı sessiz sayılır
        min_silence_duration: Minimum sessizlik süresi (saniye)

    Returns:
        List of (start_sample, end_sample) tuples
    """
    # RMS (Root Mean Square) energy hesapla
    frame_length = int(0.025 * sample_rate)  # 25ms frame
    hop_length = int(0.010 * sample_rate)    # 10ms hop

    # Energy hesapla
    energy = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i+frame_length]
        rms = np.sqrt(np.mean(frame**2))
        energy.append(rms)

    energy = np.array(energy)

    # dB'ye çevir (log scale)
    energy_db = 20 * np.log10(energy + 1e-10)  # 1e-10 ekleyerek log(0) önlenir

    # Threshold altındaki frame'leri bul
    is_silence = energy_db < threshold_db

    # Sessiz bölgeleri grupla
    silence_regions = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silence):
        if silent and not in_silence:
            # Sessizlik başladı
            silence_start = i * hop_length
            in_silence = True
        elif not silent and in_silence:
            # Sessizlik bitti
            silence_end = i * hop_length
            duration = (silence_end - silence_start) / sample_rate

            if duration >= min_silence_duration:
                silence_regions.append((silence_start, silence_end))

            in_silence = False

    # Son sessizlik devam ediyorsa
    if in_silence:
        silence_end = len(audio)
        duration = (silence_end - silence_start) / sample_rate
        if duration >= min_silence_duration:
            silence_regions.append((silence_start, silence_end))

    return silence_regions


def trim_silence(input_file: str, output_file: str,
                max_silence_duration: float = 0.5,
                threshold_db: float = -40,
                min_silence_duration: float = 0.3) -> dict:
    """
    Sessiz bölgeleri tespit edip kısalt

    Args:
        input_file: Girdi ses dosyası
        output_file: Çıktı ses dosyası
        max_silence_duration: Maximum sessizlik süresi (saniye)
        threshold_db: Sessizlik eşiği (dB)
        min_silence_duration: Tespit edilecek minimum sessizlik süresi (saniye)
    """
    print(f"[SilenceTrimmer] Loading audio: {input_file}")

    # Ses dosyasını yükle
    audio, rate = sf.read(input_file)
    original_duration = len(audio) / rate

    print(f"[SilenceTrimmer] Sample rate: {rate} Hz")
    print(f"[SilenceTrimmer] Original duration: {original_duration:.2f} seconds")
    print(f"[SilenceTrimmer] Detecting silence (threshold: {threshold_db} dB)...")

    start_time = time.time()

    # Sessiz bölgeleri tespit et
    silence_regions = detect_silence_regions(
        audio, rate, threshold_db, min_silence_duration
    )

    print(f"[SilenceTrimmer] Found {len(silence_regions)} silence regions")

    # Yeni ses oluştur
    segments = []
    last_end = 0
    total_trimmed = 0

    for i, (start, end) in enumerate(silence_regions):
        silence_duration = (end - start) / rate

        # Konuşma bölgesini ekle
        if start > last_end:
            segments.append(audio[last_end:start])

        # Sessizlik bölgesini kısalt
        if silence_duration > max_silence_duration:
            # max_silence_duration kadar sessizlik bırak
            silence_samples = int(max_silence_duration * rate)
            segments.append(audio[start:start + silence_samples])

            trimmed = silence_duration - max_silence_duration
            total_trimmed += trimmed

            print(f"[SilenceTrimmer]   Region {i+1}: {silence_duration:.2f}s -> {max_silence_duration:.2f}s "
                  f"(trimmed {trimmed:.2f}s)")
        else:
            # Kısaltmaya gerek yok
            segments.append(audio[start:end])

        last_end = end

    # Son konuşma bölgesini ekle
    if last_end < len(audio):
        segments.append(audio[last_end:])

    # Tüm segmentleri birleştir
    if segments:
        trimmed_audio = np.concatenate(segments)
    else:
        trimmed_audio = audio

    new_duration = len(trimmed_audio) / rate
    elapsed = time.time() - start_time

    print(f"[SilenceTrimmer] Processing completed in {elapsed:.2f} seconds")
    print(f"[SilenceTrimmer] New duration: {new_duration:.2f} seconds")
    print(f"[SilenceTrimmer] Total time saved: {total_trimmed:.2f} seconds ({total_trimmed/original_duration*100:.1f}%)")

    # Kaydet
    print(f"[SilenceTrimmer] Saving to: {output_file}")
    sf.write(output_file, trimmed_audio, rate)

    return {
        'input_file': input_file,
        'output_file': output_file,
        'original_duration': original_duration,
        'new_duration': new_duration,
        'time_saved': total_trimmed,
        'silence_regions_found': len(silence_regions),
        'processing_time': elapsed
    }


if __name__ == "__main__":
    # SpeechBrain çıktısını işle
    input_file = "output_tests/speechbrain_enhanced.wav"
    output_dir = Path("output_tests")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SILENCE TRIMMER TEST")
    print("=" * 60)
    print()

    # Test 1: 0.5 saniye max silence
    print("\n### Test 1: Max silence = 0.5 seconds")
    result1 = trim_silence(
        input_file,
        str(output_dir / "speechbrain_trimmed_0.5s.wav"),
        max_silence_duration=0.5,
        threshold_db=-40,
        min_silence_duration=0.3
    )

    # Test 2: 0.3 saniye max silence (daha agresif)
    print("\n### Test 2: Max silence = 0.3 seconds (more aggressive)")
    result2 = trim_silence(
        input_file,
        str(output_dir / "speechbrain_trimmed_0.3s.wav"),
        max_silence_duration=0.3,
        threshold_db=-40,
        min_silence_duration=0.3
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for i, result in enumerate([result1, result2], 1):
        print(f"\nTest {i}:")
        print(f"  Output: {result['output_file']}")
        print(f"  Original: {result['original_duration']:.2f}s")
        print(f"  Trimmed: {result['new_duration']:.2f}s")
        print(f"  Saved: {result['time_saved']:.2f}s ({result['time_saved']/result['original_duration']*100:.1f}%)")
        print(f"  Silence regions: {result['silence_regions_found']}")

    print("\n✓ Dosyaları dinleyip karşılaştırın!")
    print(f"✓ Dosyalar: {output_dir}/ klasöründe")
