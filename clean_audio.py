#!/usr/bin/env python3
"""
Gürültü - Clean Audio CLI

Basit kullanım:
    python clean_audio.py input.wav output.wav

İleri seviye:
    python clean_audio.py input.wav output.wav --engine rnnoise --max-silence 0.3
"""

import argparse
from pathlib import Path
from gurultu import AudioPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Gürültü - Audio noise reduction and enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (SpeechBrain engine, default settings)
  python clean_audio.py input.wav output.wav

  # Use RNNoise engine
  python clean_audio.py input.wav output.wav --engine rnnoise

  # More aggressive silence trimming
  python clean_audio.py input.wav output.wav --max-silence 0.3

  # Save intermediate files for debugging
  python clean_audio.py input.wav output.wav --save-intermediate

  # Skip normalization
  python clean_audio.py input.wav output.wav --no-normalize
        """
    )

    parser.add_argument('input', type=str, help='Input audio file')
    parser.add_argument('output', type=str, help='Output audio file')

    parser.add_argument(
        '--engine', '-e',
        type=str,
        choices=['speechbrain', 'rnnoise', 'noisereduce'],
        default='speechbrain',
        help='Noise reduction engine (default: speechbrain)'
    )

    parser.add_argument(
        '--max-silence', '-s',
        type=float,
        default=0.5,
        help='Maximum silence duration in seconds (default: 0.5)'
    )

    parser.add_argument(
        '--silence-threshold',
        type=float,
        default=-40,
        help='Silence detection threshold in dB (default: -40)'
    )

    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip normalization step'
    )

    parser.add_argument(
        '--reference',
        type=str,
        help='Reference file for normalization (default: use input file)'
    )

    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate processing files'
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Create pipeline
    pipeline = AudioPipeline(
        engine=args.engine,
        max_silence=args.max_silence,
        silence_threshold_db=args.silence_threshold,
        normalize=not args.no_normalize,
        reference_file=args.reference
    )

    # Process
    try:
        result = pipeline.process(
            args.input,
            args.output,
            save_intermediate=args.save_intermediate
        )
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
