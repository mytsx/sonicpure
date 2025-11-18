#!/usr/bin/env python3
"""
Test script for SpeechBrain library
Pre-trained neural network modeli ile speech enhancement
En güçlü ama en yavaş seçenek
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import time
import torch
from scipy import signal

def clean_with_speechbrain(input_file: str, output_file: str):
    """
    SpeechBrain'in pre-trained modelini kullanarak ses temizleme

    SpeechBrain MetricGAN+ veya SepFormer gibi modeller kullanabilir
    """
    try:
        from speechbrain.inference.separation import SepformerSeparation as separator
    except ImportError:
        print("ERROR: speechbrain bulunamadı!")
        print("Kurulum: pip install speechbrain")
        return None

    print(f"[SpeechBrain] Loading audio: {input_file}")

    # Ses dosyasını yükle
    data, original_rate = sf.read(input_file)
    print(f"[SpeechBrain] Original sample rate: {original_rate} Hz")
    print(f"[SpeechBrain] Duration: {len(data)/original_rate:.2f} seconds")
    print(f"[SpeechBrain] Channels: {data.shape[1] if len(data.shape) > 1 else 1}")

    # Başlangıç zamanı
    start_time = time.time()

    # MetricGAN+ modeli 16kHz için eğitilmiş
    MODEL_SAMPLE_RATE = 16000

    # Eğer dosya farklı sample rate'deyse, 16kHz'e resample et
    temp_file = None
    if original_rate != MODEL_SAMPLE_RATE:
        print(f"[SpeechBrain] Resampling {original_rate} Hz -> {MODEL_SAMPLE_RATE} Hz...")

        # Resample
        num_samples = int(len(data) * MODEL_SAMPLE_RATE / original_rate)
        data_resampled = signal.resample(data, num_samples)

        # Geçici dosya oluştur
        temp_file = Path(output_file).parent / "temp_16khz.wav"
        sf.write(temp_file, data_resampled, MODEL_SAMPLE_RATE)
        input_to_process = str(temp_file)
        print(f"[SpeechBrain] Temporary 16kHz file created")
    else:
        input_to_process = input_file

    # SpeechBrain modelini yükle (enhancement için)
    print(f"[SpeechBrain] Loading pre-trained model...")
    print(f"[SpeechBrain] (İlk çalıştırmada model indirilecek, biraz zaman alabilir)")

    # MetricGAN+ modeli (speech enhancement için)
    # Model: speechbrain/metricgan-plus-voicebank
    try:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        enhancer = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
        )

        print(f"[SpeechBrain] Model loaded!")
        print(f"[SpeechBrain] Processing audio at {MODEL_SAMPLE_RATE} Hz...")

        # Enhance audio
        enhanced = enhancer.enhance_file(input_to_process)

        # Tensor'dan numpy'a çevir
        if torch.is_tensor(enhanced):
            enhanced = enhanced.cpu().numpy()

        # Eğer batch dimension varsa kaldır
        if len(enhanced.shape) > 1:
            enhanced = enhanced.squeeze()

        # Eğer resampling yaptıysak, çıktıyı tekrar orijinal sample rate'e döndür
        if original_rate != MODEL_SAMPLE_RATE:
            print(f"[SpeechBrain] Resampling output {MODEL_SAMPLE_RATE} Hz -> {original_rate} Hz...")
            num_samples_original = int(len(enhanced) * original_rate / MODEL_SAMPLE_RATE)
            enhanced = signal.resample(enhanced, num_samples_original)

            # Geçici dosyayı sil
            if temp_file and temp_file.exists():
                temp_file.unlink()
                print(f"[SpeechBrain] Temporary file deleted")

        elapsed = time.time() - start_time
        print(f"[SpeechBrain] Processing completed in {elapsed:.2f} seconds")

        # Sonucu kaydet (orijinal sample rate'de)
        print(f"[SpeechBrain] Saving to: {output_file} at {original_rate} Hz")
        sf.write(output_file, enhanced, original_rate)

        return {
            'engine': 'SpeechBrain (MetricGAN+)',
            'input_file': input_file,
            'output_file': output_file,
            'sample_rate': original_rate,
            'processing_time': elapsed
        }

    except Exception as e:
        print(f"[SpeechBrain] ERROR: {e}")
        print(f"[SpeechBrain] Model yüklenemedi veya işlem başarısız")

        # Geçici dosyayı temizle
        if temp_file and temp_file.exists():
            temp_file.unlink()
            print(f"[SpeechBrain] Temporary file cleaned up")

        return None


if __name__ == "__main__":
    # Test dosyası
    input_wav = "tts_fbea8465-85d5-44cf-9f6d-779a1e7c31c2.wav"

    # Çıktı klasörü
    output_dir = Path("output_tests")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SPEECHBRAIN TEST")
    print("=" * 60)
    print()

    result = clean_with_speechbrain(
        input_wav,
        str(output_dir / "speechbrain_enhanced.wav")
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
        print("\nNOT: SpeechBrain daha ağır bir model kullandığı için")
        print("     diğer yöntemlerden daha yavaş ama kalite daha iyi olabilir.")
    else:
        print("\n✗ SpeechBrain testi başarısız oldu")
