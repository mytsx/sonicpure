#!/usr/bin/env python3
"""
Audio Normalizer - Ses seviyesini dinamik olarak normalize eder
Compression ve limiting ile clipping'i önler
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

def soft_knee_compressor(audio: np.ndarray, threshold: float = 0.7, ratio: float = 4.0, knee: float = 0.1) -> np.ndarray:
    """
    Soft-knee compressor - Yüksek sesleri yumuşak bir şekilde sıkıştırır

    Args:
        audio: Ses verisi [-1, 1]
        threshold: Compression başlama eşiği (0-1)
        ratio: Sıkıştırma oranı (4:1 = 4.0)
        knee: Yumuşak geçiş genişliği

    Returns:
        Compressed audio
    """
    # Absolute value al
    abs_audio = np.abs(audio)

    # Compressed output
    compressed = np.zeros_like(audio)

    for i, sample in enumerate(audio):
        abs_sample = abs_audio[i]

        if abs_sample < (threshold - knee/2):
            # Threshold altında - değişiklik yok
            compressed[i] = sample
        elif abs_sample > (threshold + knee/2):
            # Threshold üstünde - compression uygula
            # Formül: output = threshold + (input - threshold) / ratio
            excess = abs_sample - threshold
            compressed_excess = excess / ratio
            compressed[i] = np.sign(sample) * (threshold + compressed_excess)
        else:
            # Knee bölgesinde - yumuşak geçiş
            knee_start = threshold - knee/2
            knee_range = knee
            position = (abs_sample - knee_start) / knee_range  # 0-1

            # Linear interpolation between no compression and full compression
            no_comp = abs_sample
            excess = abs_sample - threshold
            full_comp = threshold + excess / ratio

            # Smooth transition
            compressed_value = no_comp + position * (full_comp - no_comp)
            compressed[i] = np.sign(sample) * compressed_value

    return compressed


def normalize_to_target_rms(audio: np.ndarray, target_rms_db: float,
                           current_rms_db: float, max_peak_db: float = -1.0) -> tuple:
    """
    Dinamik olarak RMS'i hedef seviyeye getir, compression kullanarak clipping'i önle

    Args:
        audio: Ses verisi
        target_rms_db: Hedef RMS (dBFS)
        current_rms_db: Mevcut RMS (dBFS)
        max_peak_db: Maximum peak seviyesi (clipping önlemek için)

    Returns:
        (normalized_audio, actual_rms_db, peak_db, gain_applied_db)
    """
    # Gereken gain hesapla
    required_gain_db = target_rms_db - current_rms_db
    required_gain_linear = 10 ** (required_gain_db / 20)

    print(f"[Normalizer] Required gain: {required_gain_db:+.2f} dB")

    # Mevcut peak
    current_peak = np.max(np.abs(audio))
    current_peak_db = 20 * np.log10(current_peak) if current_peak > 0 else -np.inf

    # Eğer gain uygulanırsa peak ne olur?
    predicted_peak = current_peak * required_gain_linear
    predicted_peak_db = 20 * np.log10(predicted_peak) if predicted_peak > 0 else -np.inf

    max_peak_linear = 10 ** (max_peak_db / 20)

    if predicted_peak <= max_peak_linear:
        # Clipping riski yok - direkt gain uygula
        print(f"[Normalizer] No clipping risk, applying gain directly")
        normalized = audio * required_gain_linear
        actual_gain_db = required_gain_db

    else:
        # Clipping riski var - compression stratejisi kullan
        print(f"[Normalizer] Clipping risk detected!")
        print(f"[Normalizer]   Current peak: {current_peak_db:.2f} dBFS")
        print(f"[Normalizer]   Predicted peak: {predicted_peak_db:.2f} dBFS")
        print(f"[Normalizer]   Max allowed: {max_peak_db:.2f} dBFS")
        print(f"[Normalizer] Applying soft-knee compression...")

        # Strateji: Önce compression uygula, sonra gain
        # Compression threshold: peak seviyesinin biraz altında
        # Böylece en yüksek sesleri sıkıştırıp headroom açıyoruz

        # Compression threshold hesapla (mevcut peak'in %70'i)
        compression_threshold = current_peak * 0.7
        compression_ratio = 3.0  # 3:1 compression

        # Compression uygula
        compressed = soft_knee_compressor(audio, threshold=compression_threshold,
                                         ratio=compression_ratio, knee=0.1)

        # Compression sonrası RMS hesapla
        compressed_rms = np.sqrt(np.mean(compressed**2))
        compressed_rms_db = 20 * np.log10(compressed_rms) if compressed_rms > 0 else -np.inf

        # Şimdi hedef RMS'e ulaşmak için gain hesapla
        new_gain_needed_db = target_rms_db - compressed_rms_db
        new_gain_linear = 10 ** (new_gain_needed_db / 20)

        print(f"[Normalizer] After compression, need {new_gain_needed_db:+.2f} dB more")

        # Gain uygula
        normalized = compressed * new_gain_linear

        # Final peak kontrolü
        final_peak = np.max(np.abs(normalized))
        final_peak_db = 20 * np.log10(final_peak) if final_peak > 0 else -np.inf

        if final_peak > max_peak_linear:
            # Hala limit aşıldı - hard limiter uygula
            print(f"[Normalizer] Final limiting to {max_peak_db:.2f} dBFS")
            normalized = normalized * (max_peak_linear / final_peak)

        actual_gain_db = new_gain_needed_db  # Toplam gain

    # Final değerleri hesapla
    final_rms = np.sqrt(np.mean(normalized**2))
    final_rms_db = 20 * np.log10(final_rms) if final_rms > 0 else -np.inf
    final_peak = np.max(np.abs(normalized))
    final_peak_db = 20 * np.log10(final_peak) if final_peak > 0 else -np.inf

    return normalized, final_rms_db, final_peak_db, actual_gain_db


def normalize_audio(input_file: str, output_file: str,
                   target_rms_db: Optional[float] = None,
                   reference_file: Optional[str] = None,
                   max_peak_db: float = -1.0) -> dict:
    """
    Ses dosyasını normalize et - dinamik hesaplama ile

    Args:
        input_file: Girdi dosyası
        output_file: Çıktı dosyası
        target_rms_db: Hedef RMS seviyesi (dBFS). None ise reference_file kullanılır
        reference_file: Referans dosya (RMS seviyesini buradan al)
        max_peak_db: Maximum peak seviyesi (clipping önlemek için)

    Returns:
        dict: Normalization bilgileri
    """
    print(f"[Normalizer] Loading: {input_file}")

    # Ses dosyasını yükle
    audio, rate = sf.read(input_file)

    # Mevcut RMS hesapla
    current_rms = np.sqrt(np.mean(audio**2))
    current_rms_db = 20 * np.log10(current_rms) if current_rms > 0 else -np.inf
    current_peak = np.max(np.abs(audio))
    current_peak_db = 20 * np.log10(current_peak) if current_peak > 0 else -np.inf

    print(f"[Normalizer] Current RMS: {current_rms_db:.2f} dBFS")
    print(f"[Normalizer] Current Peak: {current_peak_db:.2f} dBFS")

    # Hedef RMS belirle
    if reference_file:
        print(f"[Normalizer] Loading reference: {reference_file}")
        ref_audio, _ = sf.read(reference_file)
        ref_rms = np.sqrt(np.mean(ref_audio**2))
        target_rms_db = 20 * np.log10(ref_rms) if ref_rms > 0 else -np.inf
        print(f"[Normalizer] Reference RMS: {target_rms_db:.2f} dBFS")
    elif target_rms_db is None:
        raise ValueError("Either target_rms_db or reference_file must be provided")

    print(f"[Normalizer] Target RMS: {target_rms_db:.2f} dBFS")

    # Normalize et
    normalized, final_rms_db, final_peak_db, gain_applied_db = normalize_to_target_rms(
        audio, target_rms_db, current_rms_db, max_peak_db
    )

    # Kaydet
    print(f"[Normalizer] Final RMS: {final_rms_db:.2f} dBFS")
    print(f"[Normalizer] Final Peak: {final_peak_db:.2f} dBFS")
    print(f"[Normalizer] Saving to: {output_file}")

    sf.write(output_file, normalized, rate)

    return {
        'input_file': input_file,
        'output_file': output_file,
        'reference_file': reference_file,
        'original_rms_db': current_rms_db,
        'original_peak_db': current_peak_db,
        'target_rms_db': target_rms_db,
        'final_rms_db': final_rms_db,
        'final_peak_db': final_peak_db,
        'gain_applied_db': gain_applied_db
    }


if __name__ == "__main__":
    print("=" * 70)
    print("AUDIO NORMALIZER - DYNAMIC")
    print("=" * 70)

    # Referans dosya ile normalize et
    original_file = "tts_fbea8465-85d5-44cf-9f6d-779a1e7c31c2.wav"
    input_file = "output_tests/speechbrain_trimmed_0.5s.wav"
    output_file = "output_tests/speechbrain_trimmed_normalized.wav"

    print(f"\n### Normalizing to match reference file RMS level")

    result = normalize_audio(
        input_file,
        output_file,
        reference_file=original_file,
        max_peak_db=-1.0
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOriginal RMS: {result['original_rms_db']:.2f} dBFS")
    print(f"Target RMS: {result['target_rms_db']:.2f} dBFS")
    print(f"Final RMS: {result['final_rms_db']:.2f} dBFS")
    print(f"Difference: {result['final_rms_db'] - result['target_rms_db']:.2f} dB")
    print(f"\nOriginal Peak: {result['original_peak_db']:.2f} dBFS")
    print(f"Final Peak: {result['final_peak_db']:.2f} dBFS")
    print(f"\n✓ Normalized file: {result['output_file']}")
