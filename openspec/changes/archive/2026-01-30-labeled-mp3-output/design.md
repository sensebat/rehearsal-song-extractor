## Context

The current `export_songs()` function writes WAV files directly to an output directory with simple `Song_01.wav` naming. We need to restructure output to support labeled, organized files with MP3 encoding.

Current flow:
```
export_songs() → writes Song_01.wav, Song_02.wav to --output-dir
```

New flow:
```
export_songs() → creates {{LABEL}}/ directory
              → creates {{LABEL}}/Lossless/ subdirectory
              → writes {{LABEL}} Song 01.wav to Lossless/
              → encodes {{LABEL}} Song 01.mp3 to root
```

## Goals / Non-Goals

**Goals:**
- Add `--label` argument that controls output directory name and file prefixes
- Create structured output with Lossless subdirectory
- Encode MP3 at 320 kbps using ffmpeg
- Preserve lossless WAV files

**Non-Goals:**
- Metadata tagging (ID3 tags, etc.)
- Other audio formats (FLAC, AAC, etc.)
- Automatic label generation from filename

## Decisions

### Decision 1: Use ffmpeg for MP3 encoding

**Choice**: Use ffmpeg with libmp3lame encoder at 320 kbps CBR.

**Rationale**: ffmpeg is already a dependency for audio extraction. Using it for MP3 encoding avoids adding new dependencies.

**Command**:
```bash
ffmpeg -i input.wav -codec:a libmp3lame -b:a 320k output.mp3
```

### Decision 2: Label is optional with sensible default

**Choice**: Make `--label` optional. Without it, output goes to `songs/` with no file prefix.

**Rationale**: Simple usage (`python rse.py file.mov`) should work out of the box. Label is for users who want organized, named output.

### Decision 3: Write WAV first, then encode MP3

**Choice**: Write WAV to Lossless/, then encode WAV→MP3 to root.

**Rationale**: This ensures we always have the lossless source. If MP3 encoding fails, we still have the WAV.

## Risks / Trade-offs

**Risk: ffmpeg MP3 encoding not available**
Some systems may not have libmp3lame compiled into ffmpeg.
→ **Mitigation**: Check for encoder availability at startup, fail with clear error message.

**Risk: Doubled disk usage during export**
WAV files are large, and we're keeping both WAV and MP3.
→ **Mitigation**: This is intentional (user wants both). Could add --mp3-only flag later if needed.
