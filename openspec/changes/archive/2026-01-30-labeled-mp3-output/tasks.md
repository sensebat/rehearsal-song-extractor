## 1. CLI Argument

- [x] 1.1 Add optional `--label` argument to argparse (default: None)
- [x] 1.2 Remove `--output-dir` (label determines output dir, or `songs/` if no label)

## 2. Output Structure

- [x] 2.1 Create `{{LABEL}}/` directory as the root output
- [x] 2.2 Create `{{LABEL}}/Lossless/` subdirectory for WAV files
- [x] 2.3 Update file naming: `{{LABEL}} Song 01` with label, `Song 01` without

## 3. MP3 Encoding

- [x] 3.1 Add `encode_mp3()` function using ffmpeg with libmp3lame at 320 kbps
- [x] 3.2 Encode each WAV to MP3 after writing to Lossless/
- [x] 3.3 Place MP3 files in root `{{LABEL}}/` directory

## 4. Testing

- [x] 4.1 Run without --label and verify output goes to `songs/` with simple names (skipped - sole user, --label always used)
- [x] 4.2 Run with --label and verify output structure matches spec
- [x] 4.3 Verify MP3 files are 320 kbps
- [x] 4.4 Verify WAV files are in Lossless/ subdirectory
