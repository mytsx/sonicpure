"""
Audio Processors - Silence trimming and normalization
"""

import numpy as np
import soundfile as sf
from typing import List, Tuple, Optional, Dict


class SilenceTrimmer:
    """Silence detection and trimming"""

    def detect_silence(self, audio: np.ndarray, sample_rate: int,
                      threshold_db: float = -40,
                      min_silence_duration: float = 0.3) -> List[Tuple[int, int]]:
        """Detect silence regions"""
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)

        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            rms = np.sqrt(np.mean(frame**2))
            energy.append(rms)

        energy = np.array(energy)
        energy_db = 20 * np.log10(energy + 1e-10)

        is_silence = energy_db < threshold_db

        silence_regions = []
        in_silence = False
        silence_start = 0

        for i, silent in enumerate(is_silence):
            if silent and not in_silence:
                silence_start = i * hop_length
                in_silence = True
            elif not silent and in_silence:
                silence_end = i * hop_length
                duration = (silence_end - silence_start) / sample_rate
                if duration >= min_silence_duration:
                    silence_regions.append((silence_start, silence_end))
                in_silence = False

        if in_silence:
            silence_end = len(audio)
            duration = (silence_end - silence_start) / sample_rate
            if duration >= min_silence_duration:
                silence_regions.append((silence_start, silence_end))

        return silence_regions

    def trim(self, input_file: str, output_file: str,
            max_silence_duration: float = 0.5,
            threshold_db: float = -40,
            min_silence_duration: float = 0.3) -> Dict:
        """Trim silence in audio file"""

        print(f"[SilenceTrimmer] Loading: {input_file}")

        audio, rate = sf.read(input_file)
        original_duration = len(audio) / rate

        print(f"[SilenceTrimmer] Detecting silence (threshold: {threshold_db} dB)...")

        silence_regions = self.detect_silence(
            audio, rate, threshold_db, min_silence_duration
        )

        print(f"[SilenceTrimmer] Found {len(silence_regions)} silence regions")

        segments = []
        last_end = 0
        total_trimmed = 0

        for i, (start, end) in enumerate(silence_regions):
            silence_duration = (end - start) / rate

            if start > last_end:
                segments.append(audio[last_end:start])

            if silence_duration > max_silence_duration:
                silence_samples = int(max_silence_duration * rate)
                segments.append(audio[start:start + silence_samples])
                trimmed = silence_duration - max_silence_duration
                total_trimmed += trimmed
            else:
                segments.append(audio[start:end])

            last_end = end

        if last_end < len(audio):
            segments.append(audio[last_end:])

        if segments:
            trimmed_audio = np.concatenate(segments)
        else:
            trimmed_audio = audio

        new_duration = len(trimmed_audio) / rate

        print(f"[SilenceTrimmer] Trimmed {total_trimmed:.2f}s ({total_trimmed/original_duration*100:.1f}%)")
        print(f"[SilenceTrimmer] Saving to: {output_file}")

        sf.write(output_file, trimmed_audio, rate)

        return {
            'original_duration': original_duration,
            'new_duration': new_duration,
            'time_saved': total_trimmed,
            'silence_regions_found': len(silence_regions)
        }


class AudioNormalizer:
    """Audio level normalization with compression"""

    def soft_knee_compressor(self, audio: np.ndarray, threshold: float = 0.7,
                            ratio: float = 3.0, knee: float = 0.1) -> np.ndarray:
        """Soft-knee compressor"""
        abs_audio = np.abs(audio)
        compressed = np.zeros_like(audio)

        for i, sample in enumerate(audio):
            abs_sample = abs_audio[i]

            if abs_sample < (threshold - knee/2):
                compressed[i] = sample
            elif abs_sample > (threshold + knee/2):
                excess = abs_sample - threshold
                compressed_excess = excess / ratio
                compressed[i] = np.sign(sample) * (threshold + compressed_excess)
            else:
                knee_start = threshold - knee/2
                knee_range = knee
                position = (abs_sample - knee_start) / knee_range

                no_comp = abs_sample
                excess = abs_sample - threshold
                full_comp = threshold + excess / ratio

                compressed_value = no_comp + position * (full_comp - no_comp)
                compressed[i] = np.sign(sample) * compressed_value

        return compressed

    def normalize(self, input_file: str, output_file: str,
                 target_rms_db: Optional[float] = None,
                 reference_file: Optional[str] = None,
                 max_peak_db: float = -1.0) -> Dict:
        """Normalize audio to target RMS level"""

        print(f"[Normalizer] Loading: {input_file}")

        audio, rate = sf.read(input_file)

        current_rms = np.sqrt(np.mean(audio**2))
        current_rms_db = 20 * np.log10(current_rms) if current_rms > 0 else -np.inf
        current_peak = np.max(np.abs(audio))
        current_peak_db = 20 * np.log10(current_peak) if current_peak > 0 else -np.inf

        print(f"[Normalizer] Current RMS: {current_rms_db:.2f} dBFS")

        # Get target RMS
        if reference_file:
            print(f"[Normalizer] Using reference: {reference_file}")
            ref_audio, _ = sf.read(reference_file)
            ref_rms = np.sqrt(np.mean(ref_audio**2))
            target_rms_db = 20 * np.log10(ref_rms) if ref_rms > 0 else -np.inf
            print(f"[Normalizer] Reference RMS: {target_rms_db:.2f} dBFS")
        elif target_rms_db is None:
            raise ValueError("Either target_rms_db or reference_file required")

        # Calculate required gain
        required_gain_db = target_rms_db - current_rms_db
        required_gain_linear = 10 ** (required_gain_db / 20)

        print(f"[Normalizer] Required gain: {required_gain_db:+.2f} dB")

        # Check clipping
        predicted_peak = current_peak * required_gain_linear
        max_peak_linear = 10 ** (max_peak_db / 20)

        if predicted_peak <= max_peak_linear:
            normalized = audio * required_gain_linear
        else:
            print(f"[Normalizer] Using compression to avoid clipping...")
            compression_threshold = current_peak * 0.7
            compressed = self.soft_knee_compressor(audio, threshold=compression_threshold, ratio=3.0)

            compressed_rms = np.sqrt(np.mean(compressed**2))
            compressed_rms_db = 20 * np.log10(compressed_rms)

            new_gain_db = target_rms_db - compressed_rms_db
            new_gain_linear = 10 ** (new_gain_db / 20)

            normalized = compressed * new_gain_linear

            final_peak = np.max(np.abs(normalized))
            if final_peak > max_peak_linear:
                normalized = normalized * (max_peak_linear / final_peak)

        # Save
        print(f"[Normalizer] Saving to: {output_file}")
        sf.write(output_file, normalized, rate)

        final_rms = np.sqrt(np.mean(normalized**2))
        final_rms_db = 20 * np.log10(final_rms)
        final_peak = np.max(np.abs(normalized))
        final_peak_db = 20 * np.log10(final_peak)

        print(f"[Normalizer] Final RMS: {final_rms_db:.2f} dBFS")

        return {
            'original_rms_db': current_rms_db,
            'target_rms_db': target_rms_db,
            'final_rms_db': final_rms_db,
            'final_peak_db': final_peak_db
        }
