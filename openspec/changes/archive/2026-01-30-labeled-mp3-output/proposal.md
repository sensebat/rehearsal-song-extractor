## Why

The current output is bare WAV files in a flat directory. Users need labeled, organized output with MP3 versions for convenient sharing and playback, while retaining lossless WAV originals.

## What Changes

- Add optional `--label` CLI argument for naming output files and directory
- **With label**: Create `{{LABEL}}/` directory with prefixed files:
  - MP3 files at root: `{{LABEL}} Song 01.mp3`, etc.
  - Lossless subdirectory: `{{LABEL}}/Lossless/{{LABEL}} Song 01.wav`, etc.
- **Without label**: Create `songs/` directory with simple names:
  - MP3 files at root: `Song 01.mp3`, etc.
  - Lossless subdirectory: `songs/Lossless/Song 01.wav`, etc.
- Encode extracted songs to 320 kbps MP3
- Keep WAV originals in Lossless subdirectory

## Capabilities

### New Capabilities
- `labeled-output`: Organize output with user-provided label prefix and structured directory layout with both MP3 and lossless WAV files.

### Modified Capabilities
<!-- None -->

## Impact

- **rse.py**:
  - Add `--label` argument
  - Modify `export_songs()` to create structured output
  - Add MP3 encoding via ffmpeg
- **Dependencies**: Relies on ffmpeg (already required for audio extraction)
