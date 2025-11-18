"""
Noise Reduction Engines
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal
from typing import Dict


class BaseEngine:
    """Base class for all noise reduction engines"""

    def process(self, input_file: str, output_file: str) -> Dict:
        """Process audio file and return results"""
        raise NotImplementedError


class SpeechBrainEngine(BaseEngine):
    """SpeechBrain MetricGAN+ engine"""

    def __init__(self):
        self.MODEL_SAMPLE_RATE = 16000

    def process(self, input_file: str, output_file: str) -> Dict:
        import torch
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        print(f"[SpeechBrain] Loading: {input_file}")

        # Load audio
        data, original_rate = sf.read(input_file)
        print(f"[SpeechBrain] Sample rate: {original_rate} Hz")

        # Resample if needed
        temp_file = None
        if original_rate != self.MODEL_SAMPLE_RATE:
            print(f"[SpeechBrain] Resampling {original_rate} Hz -> {self.MODEL_SAMPLE_RATE} Hz...")
            num_samples = int(len(data) * self.MODEL_SAMPLE_RATE / original_rate)
            data_resampled = signal.resample(data, num_samples)
            temp_file = Path(output_file).parent / "temp_16khz.wav"
            sf.write(temp_file, data_resampled, self.MODEL_SAMPLE_RATE)
            input_to_process = str(temp_file)
        else:
            input_to_process = input_file

        # Load model
        print(f"[SpeechBrain] Loading model...")
        enhancer = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
        )

        # Process
        print(f"[SpeechBrain] Processing...")
        enhanced = enhancer.enhance_file(input_to_process)

        # Convert to numpy
        if torch.is_tensor(enhanced):
            enhanced = enhanced.cpu().numpy()
        if len(enhanced.shape) > 1:
            enhanced = enhanced.squeeze()

        # Resample back if needed
        if original_rate != self.MODEL_SAMPLE_RATE:
            print(f"[SpeechBrain] Resampling back to {original_rate} Hz...")
            num_samples_original = int(len(enhanced) * original_rate / self.MODEL_SAMPLE_RATE)
            enhanced = signal.resample(enhanced, num_samples_original)
            if temp_file and temp_file.exists():
                temp_file.unlink()

        # Save
        print(f"[SpeechBrain] Saving to: {output_file}")
        sf.write(output_file, enhanced, original_rate)

        return {
            'engine': 'speechbrain',
            'sample_rate': original_rate
        }


class RNNoiseEngine(BaseEngine):
    """RNNoise engine with smoothing"""

    def __init__(self, apply_smoothing: bool = True):
        self.apply_smoothing = apply_smoothing

    def process(self, input_file: str, output_file: str) -> Dict:
        from pyrnnoise import RNNoise

        print(f"[RNNoise] Loading: {input_file}")

        # Load audio
        data, rate = sf.read(input_file)
        print(f"[RNNoise] Sample rate: {rate} Hz")

        # Convert to int16
        if data.dtype == np.float32 or data.dtype == np.float64:
            data_int16 = (data * 32767).astype(np.int16)
        else:
            data_int16 = data.astype(np.int16)

        # Reshape for RNNoise
        if len(data_int16.shape) == 1:
            data_int16 = data_int16.reshape(1, -1)
        else:
            data_int16 = data_int16.T

        # Process
        print(f"[RNNoise] Processing...")
        denoiser = RNNoise(sample_rate=rate)
        denoised_chunks = []

        for speech_prob, denoised_frame in denoiser.denoise_chunk(data_int16):
            denoised_chunks.append(denoised_frame)

        # Concatenate
        reduced_noise_int16 = np.concatenate(denoised_chunks, axis=1)

        if reduced_noise_int16.shape[0] == 1:
            reduced_noise_int16 = reduced_noise_int16[0]
        else:
            reduced_noise_int16 = reduced_noise_int16.T

        # Convert to float
        reduced_noise = reduced_noise_int16.astype(np.float32) / 32767.0

        # Apply smoothing
        if self.apply_smoothing:
            print(f"[RNNoise] Applying smoothing...")
            window_length = 11
            if len(reduced_noise) < window_length:
                window_length = len(reduced_noise) if len(reduced_noise) % 2 == 1 else len(reduced_noise) - 1
            reduced_noise = signal.savgol_filter(reduced_noise, window_length, 3)

        # Save
        print(f"[RNNoise] Saving to: {output_file}")
        sf.write(output_file, reduced_noise, rate)

        return {
            'engine': 'rnnoise',
            'sample_rate': rate,
            'smoothing': self.apply_smoothing
        }


class NoiseReduceEngine(BaseEngine):
    """NoiseReduce engine"""

    def __init__(self, prop_decrease: float = 0.7, stationary: bool = True):
        self.prop_decrease = prop_decrease
        self.stationary = stationary

    def process(self, input_file: str, output_file: str) -> Dict:
        import noisereduce as nr

        print(f"[NoiseReduce] Loading: {input_file}")

        # Load audio
        data, rate = sf.read(input_file)
        print(f"[NoiseReduce] Sample rate: {rate} Hz")

        # Process
        print(f"[NoiseReduce] Processing (prop_decrease={self.prop_decrease})...")
        reduced_noise = nr.reduce_noise(
            y=data,
            sr=rate,
            stationary=self.stationary,
            prop_decrease=self.prop_decrease
        )

        # Save
        print(f"[NoiseReduce] Saving to: {output_file}")
        sf.write(output_file, reduced_noise, rate)

        return {
            'engine': 'noisereduce',
            'sample_rate': rate,
            'prop_decrease': self.prop_decrease,
            'stationary': self.stationary
        }
