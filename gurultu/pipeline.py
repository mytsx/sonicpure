"""
Audio Processing Pipeline - All-in-one audio cleaning
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict
import time

from .engines import SpeechBrainEngine
from .processors import SilenceTrimmer, AudioNormalizer


class AudioPipeline:
    """
    All-in-one audio processing pipeline

    Usage:
        pipeline = AudioPipeline()
        result = pipeline.process("input.wav", "output.wav")
    """

    def __init__(self,
                 engine: str = 'speechbrain',
                 max_silence: float = 0.5,
                 silence_threshold_db: float = -40,
                 normalize: bool = True,
                 reference_file: Optional[str] = None):
        """
        Args:
            engine: Noise reduction engine ('speechbrain', 'rnnoise', 'noisereduce')
            max_silence: Maximum silence duration in seconds
            silence_threshold_db: Silence detection threshold
            normalize: Apply normalization
            reference_file: Reference file for normalization (optional)
        """
        self.engine_name = engine
        self.max_silence = max_silence
        self.silence_threshold_db = silence_threshold_db
        self.normalize = normalize
        self.reference_file = reference_file

        # Initialize engine
        if engine == 'speechbrain':
            from .engines import SpeechBrainEngine
            self.engine = SpeechBrainEngine()
        elif engine == 'rnnoise':
            from .engines import RNNoiseEngine
            self.engine = RNNoiseEngine(apply_smoothing=True)
        elif engine == 'noisereduce':
            from .engines import NoiseReduceEngine
            self.engine = NoiseReduceEngine(prop_decrease=0.7)
        else:
            raise ValueError(f"Unknown engine: {engine}")

        # Initialize processors
        self.trimmer = SilenceTrimmer()
        self.normalizer = AudioNormalizer()

    def process(self,
                input_file: str,
                output_file: str,
                save_intermediate: bool = False) -> Dict:
        """
        Process audio file through the complete pipeline

        Args:
            input_file: Input audio file path
            output_file: Output audio file path
            save_intermediate: Save intermediate files for debugging

        Returns:
            dict: Processing results and statistics
        """
        print("=" * 70)
        print("GÜRÜLTÜ AUDIO PIPELINE")
        print("=" * 70)
        print(f"\nInput: {input_file}")
        print(f"Output: {output_file}")
        print(f"Engine: {self.engine_name}")
        print()

        output_path = Path(output_file)
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True, parents=True)

        start_time = time.time()

        # Step 1: Noise Reduction
        print("\n" + "-" * 70)
        print("STEP 1: NOISE REDUCTION")
        print("-" * 70)

        if save_intermediate:
            denoised_file = output_dir / f"{output_path.stem}_1_denoised{output_path.suffix}"
        else:
            denoised_file = output_dir / f"temp_denoised.wav"

        denoised_result = self.engine.process(input_file, str(denoised_file))

        # Step 2: Silence Trimming
        print("\n" + "-" * 70)
        print("STEP 2: SILENCE TRIMMING")
        print("-" * 70)

        if save_intermediate:
            trimmed_file = output_dir / f"{output_path.stem}_2_trimmed{output_path.suffix}"
        else:
            trimmed_file = output_dir / f"temp_trimmed.wav"

        trimmed_result = self.trimmer.trim(
            str(denoised_file),
            str(trimmed_file),
            max_silence_duration=self.max_silence,
            threshold_db=self.silence_threshold_db
        )

        # Step 3: Normalization (optional)
        if self.normalize:
            print("\n" + "-" * 70)
            print("STEP 3: NORMALIZATION")
            print("-" * 70)

            # Use input file as reference if not provided
            ref_file = self.reference_file or input_file

            normalize_result = self.normalizer.normalize(
                str(trimmed_file),
                output_file,
                reference_file=ref_file
            )
        else:
            # Just copy trimmed file to output
            import shutil
            shutil.copy(str(trimmed_file), output_file)
            normalize_result = None

        # Cleanup temporary files
        if not save_intermediate:
            if denoised_file.exists():
                denoised_file.unlink()
            if trimmed_file.exists():
                trimmed_file.unlink()

        total_time = time.time() - start_time

        # Summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED")
        print("=" * 70)

        # Load final file for stats
        final_audio, final_rate = sf.read(output_file)
        final_duration = len(final_audio) / final_rate

        # Load input for comparison
        input_audio, input_rate = sf.read(input_file)
        input_duration = len(input_audio) / input_rate

        time_saved = input_duration - final_duration

        print(f"\nOriginal duration: {input_duration:.2f}s")
        print(f"Final duration: {final_duration:.2f}s")
        print(f"Time saved: {time_saved:.2f}s ({time_saved/input_duration*100:.1f}%)")
        print(f"Processing time: {total_time:.2f}s")
        print(f"\n✓ Output saved: {output_file}")

        return {
            'input_file': input_file,
            'output_file': output_file,
            'engine': self.engine_name,
            'original_duration': input_duration,
            'final_duration': final_duration,
            'time_saved': time_saved,
            'processing_time': total_time,
            'denoised': denoised_result,
            'trimmed': trimmed_result,
            'normalized': normalize_result
        }
