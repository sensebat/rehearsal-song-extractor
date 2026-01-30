#!/usr/bin/env python3
"""
Extract individual songs from a choir rehearsal recording.

Uses ffmpeg to extract audio, then analyzes audio features to detect
music segments (vs. speech/silence), merges nearby segments, and exports
individual song files.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def extract_audio(input_path: Path, output_path: Path) -> None:
    """Extract audio from video file to high-quality WAV."""
    print(f"Extracting audio from {input_path}...")
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(input_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "44100",  # 44.1kHz sample rate
        "-ac", "2",  # Stereo
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Audio extracted to {output_path}")


def detect_music_segments(
    audio_path: Path,
    frame_length: float = 2.0,
    hop_length: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Detect music segments in audio using spectral features.

    Music (especially choir) tends to have:
    - Higher harmonic content
    - More sustained tones
    - Higher spectral flatness variation
    - Stronger low-frequency content

    Returns list of (start_time, end_time) tuples.
    """
    print("Loading audio for analysis...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(y) / sr
    print(f"Audio duration: {duration:.1f} seconds")

    # Convert frame parameters to samples
    frame_samples = int(frame_length * sr)
    hop_samples = int(hop_length * sr)

    print("Analyzing audio features...")
    features = []
    times = []

    for start in range(0, len(y) - frame_samples, hop_samples):
        frame = y[start : start + frame_samples]
        t = start / sr

        # RMS energy - music is generally louder than silence/quiet speech
        rms = np.sqrt(np.mean(frame**2))

        # Spectral centroid - music often has richer spectral content
        spec_cent = librosa.feature.spectral_centroid(y=frame, sr=sr)[0].mean()

        # Spectral rolloff - frequency below which 85% of energy is contained
        rolloff = librosa.feature.spectral_rolloff(y=frame, sr=sr)[0].mean()

        # Zero crossing rate - speech has higher ZCR than sustained singing
        zcr = librosa.feature.zero_crossing_rate(frame)[0].mean()

        # Harmonic-to-noise ratio approximation using harmonic/percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(frame)
        harmonic_ratio = np.sum(y_harmonic**2) / (np.sum(frame**2) + 1e-10)

        # Spectral flatness - music tends to be less flat (more tonal)
        flatness = librosa.feature.spectral_flatness(y=frame)[0].mean()

        features.append({
            "rms": rms,
            "spec_cent": spec_cent,
            "rolloff": rolloff,
            "zcr": zcr,
            "harmonic_ratio": harmonic_ratio,
            "flatness": flatness,
        })
        times.append(t)

    # Normalize features
    features_array = np.array([
        [f["rms"], f["spec_cent"], f["rolloff"], f["zcr"], f["harmonic_ratio"], f["flatness"]]
        for f in features
    ])

    # Normalize each feature to 0-1 range
    for i in range(features_array.shape[1]):
        col = features_array[:, i]
        min_val, max_val = np.nanmin(col), np.nanmax(col)
        if max_val > min_val:
            features_array[:, i] = (col - min_val) / (max_val - min_val)
        else:
            features_array[:, i] = 0.0

    # Replace any NaN/inf values with 0
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

    # Score each frame for "musicness"
    # High score = likely music
    # Weights: [rms, spec_cent, rolloff, zcr, harmonic_ratio, flatness]
    # Music: higher RMS, moderate centroid, higher rolloff, lower ZCR, higher harmonic, lower flatness
    weights = np.array([0.3, 0.1, 0.15, -0.2, 0.35, -0.2])
    scores = features_array @ weights

    # Normalize scores to 0-1
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    # Apply adaptive threshold based on score distribution
    # Use percentile-based threshold to handle varying recording conditions
    threshold = np.percentile(scores, 60)  # Top 40% of frames likely music
    threshold = max(threshold, 0.4)  # But at least 0.4

    print(f"Music detection threshold: {threshold:.3f}")

    # Find music segments
    is_music = scores > threshold
    segments = []
    in_segment = False
    segment_start = 0

    for i, (t, music) in enumerate(zip(times, is_music)):
        if music and not in_segment:
            segment_start = t
            in_segment = True
        elif not music and in_segment:
            segments.append((segment_start, t))
            in_segment = False

    # Close final segment if still in one
    if in_segment:
        segments.append((segment_start, times[-1] + frame_length))

    print(f"Found {len(segments)} raw music segments")
    return segments


def merge_segments(
    segments: list[tuple[float, float]],
    min_gap: float = 8.0,
    min_duration: float = 30.0,
) -> list[tuple[float, float]]:
    """
    Merge music segments separated by less than min_gap seconds.
    Filter out segments shorter than min_duration.
    """
    if not segments:
        return []

    merged = [segments[0]]

    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end < min_gap:
            # Merge with previous segment
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Filter by minimum duration
    filtered = [(s, e) for s, e in merged if e - s >= min_duration]

    print(f"After merging (gap < {min_gap}s): {len(merged)} segments")
    print(f"After filtering (duration >= {min_duration}s): {len(filtered)} segments")

    return filtered


def export_songs(
    audio_path: Path,
    segments: list[tuple[float, float]],
    output_dir: Path,
    padding: float = 0.5,
) -> None:
    """Export each segment as a separate WAV file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading full audio for export...")
    y, sr = librosa.load(audio_path, sr=None, mono=False)

    # Handle mono vs stereo
    if y.ndim == 1:
        y = y.reshape(1, -1)

    duration = y.shape[1] / sr

    for i, (start, end) in enumerate(segments, 1):
        # Add padding but clamp to valid range
        padded_start = max(0, start - padding)
        padded_end = min(duration, end + padding)

        start_sample = int(padded_start * sr)
        end_sample = int(padded_end * sr)

        segment_audio = y[:, start_sample:end_sample]

        output_path = output_dir / f"Song_{i:02d}.wav"
        sf.write(output_path, segment_audio.T, sr)

        segment_duration = padded_end - padded_start
        print(f"Exported {output_path.name}: {format_time(padded_start)} - {format_time(padded_end)} ({segment_duration:.1f}s)")


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract songs from a choir rehearsal recording"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input video/audio file (e.g., .mov, .mp4, .wav)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("songs"),
        help="Output directory for extracted songs (default: songs/)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("temp"),
        help="Temporary directory for intermediate files (default: temp/)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=8.0,
        help="Merge segments separated by less than this many seconds (default: 8)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=30.0,
        help="Minimum song duration in seconds (default: 30)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Create temp directory
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    temp_audio = args.temp_dir / "extracted_audio.wav"

    # Step 1: Extract audio
    extract_audio(args.input, temp_audio)

    # Step 2: Detect music segments
    segments = detect_music_segments(temp_audio)

    # Step 3: Merge nearby segments
    merged_segments = merge_segments(
        segments,
        min_gap=args.min_gap,
        min_duration=args.min_duration,
    )

    if not merged_segments:
        print("\nNo songs detected. Try adjusting --min-duration or --min-gap.")
        sys.exit(0)

    # Step 4: Export songs
    print(f"\nExporting {len(merged_segments)} songs...")
    export_songs(temp_audio, merged_segments, args.output_dir)

    print(f"\nDone! Songs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
