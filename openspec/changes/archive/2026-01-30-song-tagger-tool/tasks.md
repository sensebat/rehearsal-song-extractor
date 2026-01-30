## 1. YAML Template Generation (rse.py)

- [x] 1.1 Add `generate_yaml_template()` function that creates songs.yaml
- [x] 1.2 Call it after export_songs() with song count and label
- [x] 1.3 Pre-fill album with label, year with current year, tracks with placeholders

## 2. Create rst.py Tool

- [x] 2.1 Create rst.py with argparse taking YAML file path
- [x] 2.2 Add YAML parsing to read metadata
- [x] 2.3 Add file discovery to find MP3s in same directory as YAML

## 3. Validation

- [x] 3.1 Validate track count matches MP3 file count
- [x] 3.2 Exit with clear error if mismatch

## 4. Renaming

- [x] 4.1 Add `sanitize_filename()` to handle special characters
- [x] 4.2 Rename directory to `{{ARTIST}} - {{ALBUM}}/`
- [x] 4.3 Rename MP3 files to `{{TRACK}} {{TITLE}}.mp3`
- [x] 4.4 Rename WAV files in Lossless/ to match

## 5. ID3 Tagging

- [x] 5.1 Add mutagen dependency to pyproject.toml
- [x] 5.2 Apply ID3 tags (title, artist, album, track, year) to each MP3

## 6. Testing

- [x] 6.1 Run rse.py and verify songs.yaml is created
- [x] 6.2 Edit songs.yaml with test data
- [x] 6.3 Run rst.py and verify files renamed and tagged correctly
