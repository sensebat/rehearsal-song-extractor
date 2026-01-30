## Context

After rse.py extracts songs, users need to:
1. Review output and delete false positives
2. Add metadata (song names, artist, etc.)
3. Tag MP3 files for music players

This is a two-phase workflow: extraction (rse.py) → tagging (rst.py).

## Goals / Non-Goals

**Goals:**
- Generate editable YAML template after extraction
- Separate tagging tool that reads YAML and applies metadata
- Rename files with proper song titles
- Write ID3 tags compatible with Spotify/Apple Music

**Non-Goals:**
- Album art embedding (could add later)
- Audio fingerprinting or automatic song identification
- Integration with music databases

## Decisions

### Decision 1: YAML for metadata

**Choice**: Use YAML format for the metadata file.

**Rationale**: Human-readable, easy to edit in any text editor, supports lists naturally.

**Format**:
```yaml
artist: "Choir Name"
album: "2016-01-29 Generalprobe"
year: 2016
tracks:
  - "Can you feel the love tonight"
  - "Irgendwas bleibt"
  - "Hallelujah"
```

### Decision 2: Use mutagen for ID3 tagging

**Choice**: Use the `mutagen` library for MP3 tag manipulation.

**Rationale**: Well-maintained, pure Python, supports ID3v2.4, no external dependencies.

### Decision 3: File and directory naming convention

**Choice**: Rename directory to `{{ARTIST}} - {{ALBUM}}/` and files to `{{TRACK}} {{TITLE}}.mp3`

**Rationale**: Standard music library format. Artist and album in directory name, clean track listing inside.

**Examples**:
- Directory: `Choir Name - 2016-01-29 Generalprobe/`
- MP3: `01 Can you feel the love tonight.mp3`
- WAV: `Lossless/01 Can you feel the love tonight.wav`

### Decision 4: Validate track count before applying

**Choice**: Error if track count in YAML doesn't match MP3 file count.

**Rationale**: Prevents silent misalignment. User must fix false positives before tagging.

## Risks / Trade-offs

**Risk: Special characters in song titles**
Some characters may not be valid in filenames.
→ **Mitigation**: Sanitize filenames (replace `/`, `\`, `:`, etc. with safe alternatives).

**Risk: User edits YAML incorrectly**
Malformed YAML could cause errors.
→ **Mitigation**: Clear error messages pointing to the issue.
