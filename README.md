# Split Rehearsal Audio

Extract individual songs from choir rehearsal recordings using AI-powered audio classification.

## How It Works

1. **Audio Extraction**: Uses `ffmpeg` to extract audio from video files (`.mov`, `.mp4`, etc.) into WAV format.

2. **AI Classification**: Processes the audio through MIT's Audio Spectrogram Transformer (AST), a neural network trained on AudioSet to distinguish between 527 audio classes including music, speech, singing, and instruments.

3. **Segment Smoothing**: Uses weighted voting across overlapping chunks to smooth classifications and filter out short blips (like ringtones or brief musical moments).

4. **Silence Detection**: Identifies quiet gaps (room tone) between songs to find natural cut points. Choir rehearsals typically have 4-5 seconds of silence before each piece begins.

5. **Smart Merging**: Combines music segments separated by brief interruptions (conductor instructions, page turns) into complete songs.

6. **High-Quality Export**: Extracts songs at full quality (44.1kHz stereo) regardless of the lower sample rate used for analysis.

## Usage

```bash
# Basic usage
uv run python extract_songs_ai.py ~/path/to/rehearsal.mov

# With options
uv run python extract_songs_ai.py recording.mov \
    --output-dir songs/ \
    --min-gap 8 \
    --min-duration 30
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `songs/` | Where to save extracted songs |
| `--temp-dir` | `temp/` | Directory for intermediate files |
| `--min-gap` | `8.0` | Merge songs separated by less than N seconds |
| `--min-duration` | `30.0` | Minimum song length in seconds |
| `--min-music-duration` | `15.0` | Minimum music segment to consider (filters blips) |
| `--silence-threshold` | `-40.0` | Silence detection threshold in dB |

## Output

Songs are exported as `Song_01.wav`, `Song_02.wav`, etc. in the output directory.

The script prints a timeline showing detected music vs. speech regions:

```
Detected regions:
    00:00 - 02:02 [speech] (122s)
  ♪ 02:02 - 06:42 [music] (280s)
    06:42 - 07:05 [speech] (22s)
  ♪ 07:05 - 07:47 [music] (42s)
  ...
```
