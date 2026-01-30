## 1. Humming Label Fix

- [x] 1.1 Add "hum" to the music label matching in `AudioClassifier.classify_chunk()` (line ~116)
- [x] 1.2 Verify with debug_region.py that vocal beat chunks now score higher for music

## 2. Loudness Dip Detection

- [x] 2.1 Add `find_loudness_dips()` function that analyzes 1-second windows and returns timestamps where RMS drops >6 dB below neighbors
- [x] 2.2 Add `is_continuous_loud()` helper to verify audio between dip and music start is consistently loud

## 3. Boundary Extension

- [x] 3.1 Modify `find_song_boundaries()` to search backwards (up to 15s) from detected music for loudness dips
- [x] 3.2 Extend song start boundary to after the dip when found, keeping original boundary otherwise
- [x] 3.3 Add `--dip-threshold` CLI option (default 6.0 dB) for tuning sensitivity

## 4. Default Duration Update

- [x] 4.1 Change `--min-duration` default from 30 to 60 in argparse
- [x] 4.2 Update help text to reflect new default

## 5. Testing

- [x] 5.1 Run on Generalprobe.mov and verify Song_10 now starts at ~24:19-24:20 instead of 24:23+
- [x] 5.2 Verify other 9 songs still detected correctly (no regressions)
- [x] 5.3 Verify song_02 (interrupted take <60s) is now filtered out by default
