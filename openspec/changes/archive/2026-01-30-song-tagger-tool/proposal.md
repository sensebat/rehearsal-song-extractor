## Why

After extracting songs with rse.py, users need to tag MP3s with proper metadata (title, artist, album, track number) for music players and streaming services. This should be a separate manual step after reviewing and cleaning up false positives.

## What Changes

**rse.py modifications:**
- Generate an empty YAML template file alongside extracted songs
- YAML contains placeholder fields for metadata

**New rst.py tool:**
- Reads the YAML metadata file
- Renames song files with proper titles
- Tags MP3s with ID3 metadata (title, artist, album, track number, year)

**Workflow:**
1. Run `rse.py` to extract songs → creates `songs.yaml` template
2. Review output, delete false positives
3. Edit `songs.yaml` with song names and metadata
4. Run `rst.py songs.yaml` to rename and tag

## Capabilities

### New Capabilities
- `yaml-metadata-template`: Generate YAML template with metadata fields after song extraction
- `song-tagger`: Read YAML and apply metadata tags to MP3 files, renaming them with proper titles

### Modified Capabilities
<!-- None -->

## Impact

- **rse.py**: Add YAML template generation after export
- **rst.py**: New tool (~100 lines) using mutagen for ID3 tagging
- **Dependencies**: Add `mutagen` for MP3 tagging, `pyyaml` for YAML parsing (already likely installed)
