#!/usr/bin/env python3
"""
Test script for RNNoise library
RNN tabanlı noise reduction (Mozilla tarafından geliştirilmiş)
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import time

def clean_with_rnnoise(input_file: str, output_file: str):
    """
    RNNoise kullanarak ses dosyasını temizle (pyrnnoise kullanarak)

    pyrnnoise kütüphanesi RNNoise'un Python wrapper'ı
    """
    try:
        from pyrnnoise import RNNoise
    except ImportError:
        print("ERROR: pyrnnoise bulunamadı!")
        print("Kurulum: pip install pyrnnoise")
        return None

    print(f"[RNNoise] Loading audio: {input_file}")

    # Ses dosyasını yükle
    data, rate = sf.read(input_file)
    print(f"[RNNoise] Sample rate: {rate} Hz")
    print(f"[RNNoise] Duration: {len(data)/rate:.2f} seconds")
    print(f"[RNNoise] Channels: {data.shape[1] if len(data.shape) > 1 else 1}")

    # Başlangıç zamanı
    start_time = time.time()

    # RNNoise ile işle (chunk processing)
    print(f"[RNNoise] Processing audio...")
    denoiser = RNNoise(sample_rate=rate)

    # Int16'ya çevir (RNNoise int16 bekler)
    if data.dtype == np.float32 or data.dtype == np.float64:
        data_int16 = (data * 32767).astype(np.int16)
    else:
        data_int16 = data.astype(np.int16)

    # Stereo ise shape değiştir [channels, samples]
    if len(data_int16.shape) == 1:
        # Mono: [samples] -> [1, samples]
        data_int16 = data_int16.reshape(1, -1)
    else:
        # Stereo zaten [samples, channels] -> [channels, samples] transpose et
        data_int16 = data_int16.T

    print(f"[RNNoise] Data shape: {data_int16.shape}")

    # Chunk processing (frame bazında)
    denoised_chunks = []
    frame_count = 0

    for speech_prob, denoised_frame in denoiser.denoise_chunk(data_int16):
        denoised_chunks.append(denoised_frame)
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"[RNNoise] Processed {frame_count} frames...")

    # Tüm chunk'ları birleştir
    reduced_noise_int16 = np.concatenate(denoised_chunks, axis=1)

    # Mono ise tekrar düzelt
    if reduced_noise_int16.shape[0] == 1:
        reduced_noise_int16 = reduced_noise_int16[0]
    else:
        # Transpose back to [samples, channels]
        reduced_noise_int16 = reduced_noise_int16.T

    # Float'a çevir
    reduced_noise = reduced_noise_int16.astype(np.float32) / 32767.0

    elapsed = time.time() - start_time
    print(f"[RNNoise] Processing completed in {elapsed:.2f} seconds")

    # Sonucu kaydet
    print(f"[RNNoise] Saving to: {output_file}")
    sf.write(output_file, reduced_noise, rate)

    return {
        'engine': 'RNNoise',
        'input_file': input_file,
        'output_file': output_file,
        'sample_rate': rate,
        'processing_time': elapsed
    }


if __name__ == "__main__":
    # Test dosyası
    input_wav = "tts_fbea8465-85d5-44cf-9f6d-779a1e7c31c2.wav"

    # Çıktı klasörü
    output_dir = Path("output_tests")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("RNNOISE TEST")
    print("=" * 60)
    print()

    result = clean_with_rnnoise(
        input_wav,
        str(output_dir / "rnnoise_cleaned.wav")
    )

    if result:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"\nEngine: {result['engine']}")
        print(f"Output: {result['output_file']}")
        print(f"Sample rate: {result['sample_rate']} Hz")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print("\n✓ Dosyayı dinleyip karşılaştırabilirsiniz!")
