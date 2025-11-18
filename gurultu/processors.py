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
    """Audio level normalization with gentle limiting"""

    def gentle_limiter(self, audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Yumuşak limiter - sadece çok yüksek sesleri yumuşakça kısar
        Tanh-based soft clipping kullanır (analog tarzı)
        """
        # Threshold üstündeki sesleri yumuşak kırp
        limited = np.copy(audio)

        # Tanh kullanarak yumuşak limiting (analog tarzı saturation)
        # threshold üstünde olanları smooth bir şekilde kırp
        mask = np.abs(audio) > threshold

        if np.any(mask):
            # Tanh function: smooth transition
            # threshold'da 1.0, sonra yavaşça limitleniyor
            excess = audio[mask]
            sign = np.sign(excess)
            abs_excess = np.abs(excess)

            # Normalize to threshold, apply tanh, scale back
            normalized = (abs_excess - threshold) / (1.0 - threshold)
            limited_value = threshold + (1.0 - threshold) * np.tanh(normalized * 2) / 2
            limited[mask] = sign * limited_value

        return limited

    def normalize(self, input_file: str, output_file: str,
                 target_rms_db: Optional[float] = None,
                 reference_file: Optional[str] = None,
                 max_peak_db: float = -0.5,
                 gentle_mode: bool = True) -> Dict:
        """
        Yumuşak normalization - pik sesleri önler, doğal ses kalitesini korur

        Args:
            input_file: Input file
            output_file: Output file
            target_rms_db: Target RMS level
            reference_file: Reference file for RMS
            max_peak_db: Max peak limit (default: -0.5 dBFS for safety)
            gentle_mode: Use gentle limiting instead of compression
        """

        print(f"[Normalizer] Loading: {input_file}")

        audio, rate = sf.read(input_file)

        current_rms = np.sqrt(np.mean(audio**2))
        current_rms_db = 20 * np.log10(current_rms) if current_rms > 0 else -np.inf
        current_peak = np.max(np.abs(audio))
        current_peak_db = 20 * np.log10(current_peak) if current_peak > 0 else -np.inf

        print(f"[Normalizer] Current RMS: {current_rms_db:.2f} dBFS")
        print(f"[Normalizer] Current Peak: {current_peak_db:.2f} dBFS")

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

        print(f"[Normalizer] Target RMS: {target_rms_db:.2f} dBFS")
        print(f"[Normalizer] Required gain: {required_gain_db:+.2f} dB")

        # Check if gain would cause clipping
        predicted_peak = current_peak * required_gain_linear
        predicted_peak_db = 20 * np.log10(predicted_peak) if predicted_peak > 0 else -np.inf
        max_peak_linear = 10 ** (max_peak_db / 20)

        # Track actual gain applied
        actual_gain_db = required_gain_db

        if predicted_peak <= max_peak_linear:
            # No clipping risk - apply gain directly
            print(f"[Normalizer] Applying gain directly (no clipping risk)")
            normalized = audio * required_gain_linear
        else:
            # Clipping would occur
            print(f"[Normalizer] Predicted peak: {predicted_peak_db:.2f} dBFS")
            print(f"[Normalizer] Max allowed: {max_peak_db:.2f} dBFS")

            if gentle_mode:
                # Gentle approach: reduce gain to avoid clipping, then gentle limit
                print(f"[Normalizer] Using gentle limiting approach...")

                # Calculate safe gain (peak reaches max_peak_db)
                safe_gain_linear = max_peak_linear / current_peak
                safe_gain_db = 20 * np.log10(safe_gain_linear)
                actual_gain_db = safe_gain_db

                print(f"[Normalizer] Safe gain: {safe_gain_db:+.2f} dB (no clipping)")

                # Apply safe gain
                normalized = audio * safe_gain_linear

                # Apply very gentle limiting for safety (only peaks above 0.95)
                normalized = self.gentle_limiter(normalized, threshold=0.95)

                print(f"[Normalizer] Note: Final RMS will be lower than target to avoid artifacts")
            else:
                # Fallback: just limit to max_peak
                normalized = audio * required_gain_linear
                normalized = np.clip(normalized, -max_peak_linear, max_peak_linear)

        # Save
        print(f"[Normalizer] Saving to: {output_file}")
        sf.write(output_file, normalized, rate)

        final_rms = np.sqrt(np.mean(normalized**2))
        final_rms_db = 20 * np.log10(final_rms) if final_rms > 0 else -np.inf
        final_peak = np.max(np.abs(normalized))
        final_peak_db = 20 * np.log10(final_peak) if final_peak > 0 else -np.inf

        print(f"[Normalizer] Final RMS: {final_rms_db:.2f} dBFS")
        print(f"[Normalizer] Final Peak: {final_peak_db:.2f} dBFS")

        return {
            'original_rms_db': current_rms_db,
            'target_rms_db': target_rms_db,
            'final_rms_db': final_rms_db,
            'final_peak_db': final_peak_db,
            'gain_applied_db': actual_gain_db
        }
