## Why

Songs that start with vocal beats or rhythmic intros (e.g., "wan wan wan" counting) are detected late because the AI classifies these as "humming" or "speech" rather than music. The actual song boundary should be at the conductor's count-off silence, not when melodic singing begins.

Debug analysis of Song_10 (starts at 24:20) revealed:
- A clear loudness dip at 24:19 (-48 dB vs -39 dB chatter and -32 dB vocal beat)
- The vocal beat (24:20-24:23) is classified as "humming" with near-tied music/speech scores (diff < 0.05)
- Music is only confidently detected at 24:23 when harmonized singing begins

## What Changes

1. **Add "humming" to music-related labels** - The AI correctly identifies vocal beats as "humming" but we don't count this as music. One-line fix to include "hum" in the music label matching.

2. **Add loudness-dip boundary detection** - Instead of requiring continuous silence (which misses count-offs where the conductor whispers "1, 2, 3, and..."), detect 1-second windows where RMS drops significantly below neighbors. This catches the count-off moment reliably.

3. **Increase default minimum song duration** - Change from 30s to 60s to filter interrupted attempts (user feedback: song_02 was a false positive from an interrupted take).

## Capabilities

### New Capabilities
- `loudness-dip-detection`: Detect song boundaries using coarse loudness analysis (1-second windows) to find count-off moments that aren't true silence but are significantly quieter than surrounding audio.

### Modified Capabilities
<!-- None - no existing specs yet -->

## Impact

- **rse.py**:
  - `AudioClassifier.classify_chunk()`: Add "hum" to music label matching
  - `find_song_boundaries()`: Add loudness-dip detection as fallback/complement to silence detection
  - `main()`: Change `--min-duration` default from 30 to 60
- **Regression risk**: Low - changes are additive (new detection method) or expand existing behavior (humming → music)
- **Testing**: Re-run on same rehearsal recording, verify Song_10 now starts at 24:19-24:20 instead of 24:23+
