#!/usr/bin/env python3
"""
AI-Powered Song Extraction from Choir Rehearsal Recordings.

Uses a pre-trained Hugging Face audio classification model to distinguish
music from speech, combined with silence detection to find natural cut points.

Key features:
- AI-based music/speech classification (more accurate than heuristics)
- Silence gap detection for precise cut points
- Robust against short blips (ringtones, brief musical moments)
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class TeeLogger:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, log_path: Path):
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str) -> int:
        self.stdout.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self) -> None:
        self.stdout.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def extract_audio(input_path: Path, output_path: Path) -> None:
    """Extract audio from video file to high-quality WAV."""
    print(f"Extracting audio from {input_path}...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz for model compatibility
        "-ac", "1",  # Mono for analysis
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Audio extracted to {output_path}")


class AudioClassifier:
    """Wrapper for Hugging Face audio classification model."""

    # Labels that indicate music
    MUSIC_LABELS = {
        "music", "singing", "song", "choir", "choral", "vocal_music",
        "musical_instrument", "orchestra", "classical_music",
    }

    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        print(f"Loading AI model: {model_name}")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get label mapping
        self.id2label = self.model.config.id2label
        print(f"Model loaded with {len(self.id2label)} classes")

    def classify_chunk(self, audio: np.ndarray, sr: int) -> tuple[str, float, dict]:
        """
        Classify an audio chunk as music or speech.

        Returns: (label, confidence, top_predictions)
        """
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Get top predictions
        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)
        top_preds = {
            self.id2label[idx.item()]: prob.item()
            for idx, prob in zip(top_indices, top_probs)
        }

        # Calculate music score (sum of probabilities for music-related labels)
        music_score = sum(
            probs[i].item()
            for i, label in self.id2label.items()
            if any(m in label.lower() for m in ["music", "sing", "choir", "vocal", "instrument", "hum"])
        )

        # Calculate speech score
        speech_score = sum(
            probs[i].item()
            for i, label in self.id2label.items()
            if any(s in label.lower() for s in ["speech", "talk", "conversation", "narration"])
        )

        if music_score > speech_score:
            return "music", music_score, top_preds
        else:
            return "speech", speech_score, top_preds


def is_continuous_loud(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    threshold_db: float = -45,
    max_gap_sec: float = 0.5,
) -> bool:
    """
    Check if audio is continuously loud between two timestamps.

    Used to verify that a loudness dip leads into continuous audio
    (not just sporadic sounds) before detected music.

    Returns True if audio is mostly above threshold with no gaps > max_gap_sec.
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    region = audio[start_sample:end_sample]

    if len(region) == 0:
        return False

    # Analyze in small windows
    window_samples = int(0.1 * sr)  # 100ms windows
    n_windows = len(region) // window_samples

    if n_windows == 0:
        return True  # Very short region, assume it's fine

    # Compute RMS for each window
    loud_windows = []
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        chunk = region[start:end]
        rms = np.sqrt(np.mean(chunk ** 2))
        db = 20 * np.log10(rms + 1e-10)
        loud_windows.append(db >= threshold_db)

    # Check for gaps (consecutive quiet windows)
    max_gap_windows = int(max_gap_sec / 0.1)
    consecutive_quiet = 0

    for is_loud in loud_windows:
        if not is_loud:
            consecutive_quiet += 1
            if consecutive_quiet > max_gap_windows:
                return False
        else:
            consecutive_quiet = 0

    return True


def find_loudness_dips(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 1.0,
    dip_threshold_db: float = 6.0,
) -> list[tuple[float, float]]:
    """
    Find timestamps where loudness dips significantly below neighbors.

    Detects count-off moments where the conductor's whispered count creates
    a relative quiet moment (not absolute silence).

    Returns list of (dip_start, dip_end) tuples for each detected dip.
    """
    window_samples = int(window_sec * sr)
    n_windows = len(audio) // window_samples

    if n_windows < 3:
        return []

    # Compute RMS for each window
    rms_values = []
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        chunk = audio[start:end]
        rms = np.sqrt(np.mean(chunk ** 2))
        rms_values.append(rms)

    # Convert to dB
    rms_values = np.array(rms_values)
    db_values = 20 * np.log10(rms_values + 1e-10)

    # Find dips: windows that are significantly quieter than both neighbors
    dips = []
    for i in range(1, n_windows - 1):
        prev_db = db_values[i - 1]
        curr_db = db_values[i]
        next_db = db_values[i + 1]

        if curr_db < prev_db - dip_threshold_db and curr_db < next_db - dip_threshold_db:
            dip_start = i * window_sec
            dip_end = (i + 1) * window_sec
            dips.append((dip_start, dip_end))

    return dips


def detect_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40,
    min_silence_duration: float = 2.0,
) -> list[tuple[float, float]]:
    """
    Detect silence regions in audio.

    Returns list of (start, end) tuples for silence regions.
    """
    # Convert to dB
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)

    # Find frames below threshold
    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop_length)

    is_silent = db < threshold_db
    silence_regions = []

    in_silence = False
    silence_start = 0

    for i, (t, silent) in enumerate(zip(times, is_silent)):
        if silent and not in_silence:
            silence_start = t
            in_silence = True
        elif not silent and in_silence:
            if t - silence_start >= min_silence_duration:
                silence_regions.append((silence_start, t))
            in_silence = False

    # Close final silence if still in one
    if in_silence and times[-1] - silence_start >= min_silence_duration:
        silence_regions.append((silence_start, times[-1]))

    return silence_regions


def classify_audio_segments(
    audio_path: Path,
    chunk_duration: float = 5.0,
    hop_duration: float = 2.5,
) -> list[Segment]:
    """
    Classify audio in overlapping chunks using AI model.

    Returns list of Segments with music/speech labels.
    """
    print("Loading audio for AI classification...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(y) / sr
    print(f"Audio duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    classifier = AudioClassifier()

    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(hop_duration * sr)

    segments = []
    total_chunks = (len(y) - chunk_samples) // hop_samples + 1

    print(f"Classifying {total_chunks} audio chunks...")

    for i, start in enumerate(range(0, len(y) - chunk_samples, hop_samples)):
        chunk = y[start : start + chunk_samples]
        t_start = start / sr
        t_end = (start + chunk_samples) / sr

        label, confidence, top_preds = classifier.classify_chunk(chunk, sr)

        segments.append(Segment(
            start=t_start,
            end=t_end,
            label=label,
            confidence=confidence,
        ))

        if (i + 1) % 50 == 0 or i == total_chunks - 1:
            print(f"  Processed {i + 1}/{total_chunks} chunks ({100*(i+1)/total_chunks:.0f}%)")

    return segments


def smooth_labels(
    segments: list[Segment],
    min_duration: float = 10.0,
) -> list[tuple[float, float, str]]:
    """
    Smooth segment labels to remove short blips.

    Merges consecutive segments with same label and filters out
    segments shorter than min_duration.
    """
    if not segments:
        return []

    # Vote on each time point
    time_step = 0.5
    max_time = max(s.end for s in segments)
    time_points = np.arange(0, max_time, time_step)

    labels = []
    for t in time_points:
        # Find all segments covering this time
        covering = [s for s in segments if s.start <= t < s.end]
        if covering:
            # Weighted vote by confidence
            music_weight = sum(s.confidence for s in covering if s.label == "music")
            speech_weight = sum(s.confidence for s in covering if s.label == "speech")
            labels.append("music" if music_weight > speech_weight else "speech")
        else:
            labels.append("speech")

    # Merge consecutive same labels
    merged = []
    current_label = labels[0]
    current_start = 0

    for i, label in enumerate(labels[1:], 1):
        if label != current_label:
            merged.append((time_points[current_start], time_points[i - 1] + time_step, current_label))
            current_label = label
            current_start = i

    merged.append((time_points[current_start], time_points[-1] + time_step, current_label))

    # Filter short segments (removes blips)
    filtered = [(s, e, l) for s, e, l in merged if e - s >= min_duration or l == "speech"]

    # Re-merge after filtering
    final = []
    for start, end, label in filtered:
        if final and final[-1][2] == label:
            final[-1] = (final[-1][0], end, label)
        else:
            final.append((start, end, label))

    return final


def find_song_boundaries(
    audio_path: Path,
    music_segments: list[tuple[float, float, str]],
    silence_threshold_db: float = -40,
    pre_song_silence: float = 3.0,
    dip_threshold_db: float = 6.0,
    max_lookback: float = 15.0,
) -> list[tuple[float, float]]:
    """
    Find precise song boundaries using silence and loudness dip detection.

    Looks for silence gaps before music segments to find natural cut points.
    Also searches backwards from detected music for loudness dips (count-off moments)
    to capture vocal intros that may be misclassified as speech.
    """
    print("\nDetecting silence regions for precise boundaries...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    silence_regions = detect_silence(
        y, sr,
        threshold_db=silence_threshold_db,
        min_silence_duration=pre_song_silence,
    )

    print(f"Found {len(silence_regions)} silence regions (>= {pre_song_silence}s)")

    # Also detect loudness dips for count-off detection
    loudness_dips = find_loudness_dips(y, sr, dip_threshold_db=dip_threshold_db)
    print(f"Found {len(loudness_dips)} loudness dips (>= {dip_threshold_db} dB drop)")

    # Extract only music segments
    music_only = [(s, e) for s, e, l in music_segments if l == "music"]

    if not music_only:
        return []

    # For each music segment, find the best start/end points
    refined_songs = []

    for music_start, music_end in music_only:
        # Find silence just before this music segment
        best_start = music_start
        found_silence = False

        for sil_start, sil_end in silence_regions:
            # Silence that ends close to (or slightly into) the music start
            if sil_end >= music_start - 5 and sil_end <= music_start + 10:
                # Start from end of silence (beginning of music)
                best_start = sil_end
                found_silence = True
                break
            # Silence that's within the first part of music (conductor cue)
            if sil_start >= music_start and sil_end <= music_start + 15:
                best_start = sil_end
                found_silence = True

        # If no silence found, look for loudness dips (count-off detection)
        if not found_silence:
            search_start = max(0, music_start - max_lookback)

            # Find dips in the lookback window, closest to music_start first
            candidate_dips = [
                (dip_start, dip_end)
                for dip_start, dip_end in loudness_dips
                if dip_end >= search_start and dip_end <= music_start + 5
            ]

            # Sort by proximity to music_start (closest first)
            candidate_dips.sort(key=lambda d: abs(d[1] - music_start))

            for dip_start, dip_end in candidate_dips:
                # Verify continuous loud audio from dip to music
                if is_continuous_loud(y, sr, dip_end, music_start):
                    best_start = dip_end
                    print(f"  Extended song boundary from {music_start:.1f}s to {best_start:.1f}s via loudness dip")
                    break

        # Find silence just after this music segment
        best_end = music_end
        for sil_start, sil_end in silence_regions:
            # Silence that starts close to music end
            if sil_start >= music_end - 10 and sil_start <= music_end + 5:
                best_end = sil_start
                break

        refined_songs.append((best_start, best_end))

    return refined_songs


def merge_nearby_songs(
    songs: list[tuple[float, float]],
    max_gap: float = 8.0,
    min_duration: float = 30.0,
) -> list[tuple[float, float]]:
    """
    Merge songs separated by small gaps and filter short ones.
    """
    if not songs:
        return []

    merged = [songs[0]]

    for start, end in songs[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end < max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Filter by duration
    filtered = [(s, e) for s, e in merged if e - s >= min_duration]

    print(f"\nAfter merging (gap < {max_gap}s): {len(merged)} songs")
    print(f"After filtering (>= {min_duration}s): {len(filtered)} songs")

    return filtered


def encode_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "320k") -> bool:
    """Encode WAV to MP3 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def export_songs(
    audio_path: Path,
    songs: list[tuple[float, float]],
    label: str | None = None,
    original_audio_path: Path | None = None,
) -> None:
    """Export songs as WAV and MP3 files with optional label prefix."""
    # Determine output directory and file prefix
    if label:
        output_dir = Path(label)
        file_prefix = f"{label} "
    else:
        output_dir = Path("songs")
        file_prefix = ""

    lossless_dir = output_dir / "Lossless"
    output_dir.mkdir(parents=True, exist_ok=True)
    lossless_dir.mkdir(parents=True, exist_ok=True)

    # Use original high-quality audio if available
    source_path = original_audio_path or audio_path

    print(f"\nLoading audio for export from {source_path}...")
    y, sr = librosa.load(source_path, sr=None, mono=False)

    if y.ndim == 1:
        y = y.reshape(1, -1)

    duration = y.shape[1] / sr

    print(f"\nExporting {len(songs)} songs:")
    for i, (start, end) in enumerate(songs, 1):
        # Clamp to valid range
        start = max(0, start)
        end = min(duration, end)

        start_sample = int(start * sr)
        end_sample = int(end * sr)

        segment = y[:, start_sample:end_sample]

        # File names
        base_name = f"{file_prefix}Song {i:02d}"
        wav_path = lossless_dir / f"{base_name}.wav"
        mp3_path = output_dir / f"{base_name}.mp3"

        # Write WAV to Lossless/
        sf.write(wav_path, segment.T, sr)

        # Encode MP3 to root
        if encode_mp3(wav_path, mp3_path):
            print(f"  {base_name}: {format_time(start)} - {format_time(end)} ({end - start:.0f}s)")
        else:
            print(f"  {base_name}: WAV saved, MP3 encoding failed")


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered song extraction from choir rehearsals"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input video/audio file",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("temp"),
        help="Temp directory (default: temp/)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=8.0,
        help="Merge songs separated by less than N seconds (default: 8)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=60.0,
        help="Minimum song duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--min-music-duration",
        type=float,
        default=15.0,
        help="Minimum music segment to consider (filters blips, default: 15)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-40.0,
        help="Silence threshold in dB (default: -40)",
    )
    parser.add_argument(
        "--dip-threshold",
        type=float,
        default=6.0,
        help="Loudness dip threshold in dB for count-off detection (default: 6)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for output files and directory (e.g., '2016-01-29 Generalprobe')",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory and set up logging
    if args.label:
        output_dir = Path(args.label)
        log_path = output_dir / f"{args.label}.log"
    else:
        output_dir = Path("songs")
        log_path = output_dir / "extraction.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(log_path)
    sys.stdout = logger

    # Create temp directory
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    temp_audio = args.temp_dir / "extracted_audio_16k.wav"
    temp_audio_hq = args.temp_dir / "extracted_audio_hq.wav"

    # Step 1: Extract audio (16kHz mono for analysis)
    extract_audio(args.input, temp_audio)

    # Also extract high-quality version for export
    print("\nExtracting high-quality audio for export...")
    cmd = [
        "ffmpeg", "-y", "-i", str(args.input),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(temp_audio_hq),
    ]
    subprocess.run(cmd, capture_output=True)

    # Step 2: AI classification
    segments = classify_audio_segments(temp_audio)

    # Step 3: Smooth and filter labels
    print("\nSmoothing classifications and filtering short segments...")
    smoothed = smooth_labels(segments, min_duration=args.min_music_duration)

    music_count = sum(1 for _, _, l in smoothed if l == "music")
    print(f"Found {music_count} music regions after smoothing")

    # Print detected regions
    print("\nDetected regions:")
    for start, end, label in smoothed:
        marker = "♪" if label == "music" else " "
        print(f"  {marker} {format_time(start)} - {format_time(end)} [{label}] ({end - start:.0f}s)")

    # Step 4: Find precise boundaries using silence and loudness dips
    songs = find_song_boundaries(
        temp_audio,
        smoothed,
        silence_threshold_db=args.silence_threshold,
        dip_threshold_db=args.dip_threshold,
    )

    # Step 5: Merge and filter
    final_songs = merge_nearby_songs(
        songs,
        max_gap=args.min_gap,
        min_duration=args.min_duration,
    )

    if not final_songs:
        print("\nNo songs detected. Try adjusting parameters.")
        sys.exit(0)

    # Step 6: Export using high-quality audio
    export_songs(temp_audio_hq, final_songs, label=args.label)

    print(f"\nDone! Songs saved to {output_dir}/")
    print(f"Log saved to {log_path}")

    # Close logger
    logger.close()
    sys.stdout = logger.stdout


if __name__ == "__main__":
    main()
