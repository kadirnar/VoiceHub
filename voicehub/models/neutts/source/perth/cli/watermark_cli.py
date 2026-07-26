#!/usr/bin/env python
"""
Command line interface for Perth watermarking.
"""

import argparse
import os
import sys
import numpy as np
from typing import Optional, List

from voicehub.models.neutts.source.perth.perth_net.perth_net_implicit.perth_watermarker import (
    PerthImplicitWatermarker,
)
from voicehub.models.neutts.source.perth.dummy_watermarker import DummyWatermarker
from voicehub.models.neutts.source.perth.config import get_config
from voicehub.models.neutts.source.perth.utils import (
    load_audio,
    save_audio,
    calculate_audio_metrics,
    plot_audio_comparison,
)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perth - Audio Watermarking Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the output watermarked audio file. "
        "If not provided, appends '_watermarked' to the input filename",
    )
    parser.add_argument(
        "--method", "-m", choices=["perth", "dummy"], help="Watermarking method to use"
    )
    parser.add_argument(
        "--extract",
        "-e",
        action="store_true",
        help="Extract watermark from the input file instead of applying a watermark",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "cuda"],
        help="Device to use for neural network processing",
    )
    parser.add_argument("--config", "-c", help="Path to a configuration file")
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Generate visualization of watermark effect (only when not extracting)",
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main function for the watermarking CLI."""
    parsed_args = parse_args(args)

    # Load configuration
    config = get_config(parsed_args.config)

    # Override config with command line arguments if provided
    if parsed_args.method:
        config.set("general", "default_watermarker", parsed_args.method)
    if parsed_args.device:
        config.set("perth", "device", parsed_args.device)

    method = config.get("general", "default_watermarker")
    device = config.get("perth", "device")

    try:
        # Load audio file
        print(f"Loading audio file: {parsed_args.input_file}")
        wav, sr = load_audio(parsed_args.input_file)

        # Initialize watermarker
        if method == "perth":
            print(f"Initializing Perth watermarker (device: {device})...")
            models_dir = config.get("perth", "models_dir")
            run_name = config.get("perth", "run_name")
            watermarker = PerthImplicitWatermarker(
                run_name=run_name, models_dir=models_dir, device=device
            )
        else:
            print("Initializing dummy watermarker...")
            watermarker = DummyWatermarker()

        if parsed_args.extract:
            # Extract watermark
            print("Extracting watermark...")
            watermark = watermarker.get_watermark(wav, sample_rate=sr)
            print(f"Extracted watermark: {watermark}")
            print(f"Watermark confidence: {np.mean(watermark):.4f}")
            return 0
        else:
            # Apply watermark
            print("Applying watermark...")
            original_audio = wav.copy()  # Save original for comparison
            watermarked_audio = watermarker.apply_watermark(
                wav, watermark=None, sample_rate=sr
            )

            # Save watermarked audio
            if parsed_args.output:
                output_path = parsed_args.output
            else:
                base, ext = os.path.splitext(parsed_args.input_file)
                output_path = f"{base}_watermarked{ext}"

            save_audio(watermarked_audio, output_path, sr)
            print(f"Watermarked audio saved to: {output_path}")

            # Verify watermark
            print("Verifying watermark...")
            extracted = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
            print(f"Watermark verification confidence: {np.mean(extracted):.4f}")

            # Calculate and display quality metrics
            metrics = calculate_audio_metrics(original_audio, watermarked_audio)
            print("\nAudio Quality Metrics:")
            print(f"  Signal-to-Noise Ratio (SNR): {metrics['snr']:.2f} dB")
            print(f"  Mean Squared Error (MSE): {metrics['mse']:.8f}")
            print(f"  Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr']:.2f} dB")

            # Generate visualization if requested
            if parsed_args.visualize:
                viz_path = os.path.splitext(output_path)[0] + "_comparison.png"
                print(f"\nGenerating visualization to: {viz_path}")
                plot_audio_comparison(original_audio, watermarked_audio, sr, viz_path)

            return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
