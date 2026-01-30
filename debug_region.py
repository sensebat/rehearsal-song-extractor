#!/usr/bin/env python3
"""
Debug tool to analyze a specific time region in the audio.

Shows:
- Silence detection at various thresholds
- AI classification scores for chunks in the region
- RMS energy levels (loudness)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.s"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def parse_time(time_str: str) -> float:
    """Parse MM:SS or MM:SS.s to seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    raise ValueError(f"Invalid time format: {time_str}. Use MM:SS or MM:SS.s")


def detect_silences_multi_threshold(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    db_threshold: float = -40,
) -> dict:
    """
    Detect silences with multiple duration thresholds.
    Returns dict mapping min_duration -> list of (start, end) tuples.
    """
    # Extract region
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    region = audio[start_sample:end_sample]

    # Compute RMS in small frames
    rms = librosa.feature.rms(y=region, frame_length=1024, hop_length=256)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)

    hop_length = 256
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop_length)
    times = times + start_time  # Offset to absolute time

    is_silent = db < db_threshold

    # Find all silence regions (no minimum duration filter yet)
    raw_silences = []
    in_silence = False
    silence_start = 0

    for i, (t, silent) in enumerate(zip(times, is_silent)):
        if silent and not in_silence:
            silence_start = t
            in_silence = True
        elif not silent and in_silence:
            raw_silences.append((silence_start, t, t - silence_start))
            in_silence = False

    if in_silence:
        raw_silences.append((silence_start, times[-1], times[-1] - silence_start))

    # Group by duration threshold
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    result = {}
    for thresh in thresholds:
        result[thresh] = [(s, e) for s, e, d in raw_silences if d >= thresh]

    return result, raw_silences, times, db


def classify_region(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    chunk_duration: float = 3.0,
    hop_duration: float = 1.0,
) -> list:
    """
    Classify audio chunks in a region and return detailed scores.
    """
    print("Loading AI model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    id2label = model.config.id2label

    # Extract region
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    region = audio[start_sample:end_sample]

    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(hop_duration * sr)

    results = []

    for start in range(0, len(region) - chunk_samples, hop_samples):
        chunk = region[start : start + chunk_samples]
        t_start = start_time + start / sr
        t_end = t_start + chunk_duration

        # Classify
        inputs = feature_extractor(
            chunk, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Calculate music vs speech scores
        music_score = sum(
            probs[i].item()
            for i, label in id2label.items()
            if any(m in label.lower() for m in ["music", "sing", "choir", "vocal", "instrument", "hum"])
        )
        speech_score = sum(
            probs[i].item()
            for i, label in id2label.items()
            if any(s in label.lower() for s in ["speech", "talk", "conversation", "narration"])
        )

        # Get top 5 predictions
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)
        top_preds = [
            (id2label[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

        results.append({
            "start": t_start,
            "end": t_end,
            "music_score": music_score,
            "speech_score": speech_score,
            "label": "music" if music_score > speech_score else "speech",
            "top_preds": top_preds,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Debug audio analysis for a specific time region"
    )
    parser.add_argument("input", type=Path, help="Input audio/video file")
    parser.add_argument("start", type=str, help="Start time (MM:SS or MM:SS.s)")
    parser.add_argument("end", type=str, help="End time (MM:SS or MM:SS.s)")
    parser.add_argument(
        "--silence-db", type=float, default=-40, help="Silence threshold in dB (default: -40)"
    )
    parser.add_argument(
        "--no-classify", action="store_true", help="Skip AI classification (faster)"
    )
    parser.add_argument(
        "--chunk-size", type=float, default=3.0, help="Classification chunk size in seconds"
    )
    parser.add_argument(
        "--chunk-hop", type=float, default=1.0, help="Classification hop size in seconds"
    )

    args = parser.parse_args()

    start_time = parse_time(args.start)
    end_time = parse_time(args.end)

    print(f"Analyzing region: {format_time(start_time)} - {format_time(end_time)}")
    print(f"Duration: {end_time - start_time:.1f} seconds")
    print()

    # Load or extract audio
    temp_audio = Path("temp/debug_audio_16k.wav")
    if not temp_audio.exists() or not args.input.suffix == ".wav":
        temp_audio.parent.mkdir(parents=True, exist_ok=True)
        print(f"Extracting audio from {args.input}...")
        cmd = [
            "ffmpeg", "-y", "-i", str(args.input),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(temp_audio),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}", file=sys.stderr)
            sys.exit(1)

    print("Loading audio...")
    y, sr = librosa.load(temp_audio, sr=16000, mono=True)
    duration = len(y) / sr
    print(f"Total duration: {format_time(duration)}")
    print()

    # Silence analysis
    print("=" * 60)
    print("SILENCE ANALYSIS")
    print("=" * 60)
    print(f"Threshold: {args.silence_db} dB")
    print()

    silences_by_thresh, raw_silences, times, db = detect_silences_multi_threshold(
        y, sr, start_time, end_time, db_threshold=args.silence_db
    )

    print("All detected quiet regions (no duration filter):")
    print("-" * 40)
    for sil_start, sil_end, sil_dur in sorted(raw_silences):
        print(f"  {format_time(sil_start)} - {format_time(sil_end)} ({sil_dur:.2f}s)")
    print()

    print("Silences by minimum duration threshold:")
    print("-" * 40)
    for thresh in sorted(silences_by_thresh.keys()):
        silences = silences_by_thresh[thresh]
        print(f"  >= {thresh:.1f}s: {len(silences)} silence(s)")
        for s, e in silences:
            print(f"           {format_time(s)} - {format_time(e)} ({e-s:.2f}s)")
    print()

    # Loudness profile
    print("=" * 60)
    print("LOUDNESS PROFILE (1-second windows)")
    print("=" * 60)

    window_size = int(sr * 1.0)
    for t in np.arange(start_time, end_time, 1.0):
        s = int(t * sr)
        e = min(s + window_size, len(y))
        chunk = y[s:e]
        rms = np.sqrt(np.mean(chunk ** 2))
        db_val = 20 * np.log10(rms + 1e-10)
        bar_len = max(0, int((db_val + 60) / 2))  # Scale -60 to 0 dB
        bar = "#" * bar_len
        silence_marker = " [SILENCE]" if db_val < args.silence_db else ""
        print(f"  {format_time(t)}: {db_val:6.1f} dB |{bar:<30}|{silence_marker}")
    print()

    # AI Classification
    if not args.no_classify:
        print("=" * 60)
        print("AI CLASSIFICATION")
        print("=" * 60)
        print(f"Chunk size: {args.chunk_size}s, hop: {args.chunk_hop}s")
        print()

        results = classify_region(
            y, sr, start_time, end_time,
            chunk_duration=args.chunk_size,
            hop_duration=args.chunk_hop,
        )

        print("Classification results:")
        print("-" * 70)
        for r in results:
            label_marker = "M" if r["label"] == "music" else "S"
            diff = r["music_score"] - r["speech_score"]
            diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
            print(
                f"  {format_time(r['start'])}-{format_time(r['end'])}: "
                f"[{label_marker}] music={r['music_score']:.3f} speech={r['speech_score']:.3f} "
                f"(diff={diff_str})"
            )
            # Show top predictions
            top_str = ", ".join(f"{label}:{prob:.2f}" for label, prob in r["top_preds"][:3])
            print(f"           Top: {top_str}")
        print()

        # Summary
        music_chunks = sum(1 for r in results if r["label"] == "music")
        speech_chunks = sum(1 for r in results if r["label"] == "speech")
        print(f"Summary: {music_chunks} music chunks, {speech_chunks} speech chunks")

        # Find transitions
        print()
        print("Transitions (speech -> music or music -> speech):")
        print("-" * 40)
        for i in range(1, len(results)):
            if results[i]["label"] != results[i-1]["label"]:
                print(
                    f"  {format_time(results[i]['start'])}: "
                    f"{results[i-1]['label']} -> {results[i]['label']}"
                )

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
