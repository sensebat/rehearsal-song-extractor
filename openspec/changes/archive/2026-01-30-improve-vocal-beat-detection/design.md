## Context

The song extraction tool uses an AI audio classifier (MIT/ast-finetuned-audioset-10-10-0.4593) to distinguish music from speech, then refines boundaries using silence detection. The current approach fails when:

1. Songs start with non-melodic vocal elements (beats, counts, rhythmic speaking)
2. The count-off gap is too short for silence detection (< 3 seconds)
3. The conductor's whispered count ("1, 2, 3, and...") prevents true silence

Current detection flow:
```
Audio → AI Classification (5s chunks) → Smooth Labels → Silence-based Boundaries → Merge/Filter
```

The AI correctly identifies vocal beats as "humming" but our scoring doesn't count humming as music-related.

## Goals / Non-Goals

**Goals:**
- Detect song starts at the count-off moment, not when melodic singing begins
- Handle vocal beat intros ("wan wan wan") that precede actual singing
- Reduce false positives from interrupted takes (< 60s)
- Maintain accuracy on the other 9 correctly-detected songs

**Non-Goals:**
- Improving the underlying AI model or fine-tuning it
- Handling songs that start without any pause (cross-fade scenarios)
- Real-time processing or streaming analysis

## Decisions

### Decision 1: Add "hum" to music label matching

**Choice**: Include "hum" in the music score calculation alongside "music", "sing", "choir", "vocal", "instrument".

**Rationale**: The AI model correctly identifies vocal beats as "humming" (scores of 0.25-0.38 in the problematic region). This is semantically correct - rhythmic vocal sounds are humming. We're not fixing a model error, just adjusting what we consider "music-adjacent".

**Alternatives considered**:
- Lower the music/speech threshold → Too risky, could cause false positives elsewhere
- Add a separate "humming" category → Overcomplicates the binary music/speech distinction

### Decision 2: Loudness-dip detection using 1-second windows

**Choice**: Complement silence detection with coarse loudness analysis that finds windows where RMS drops significantly (e.g., > 6 dB) below neighboring windows.

**Rationale**: The count-off at 24:19 shows -48 dB vs -39 dB (chatter) and -32 dB (vocal beat). This 9-16 dB dip is clearly detectable even though frame-level silence detection only finds 0.11s of true quiet. The conductor's whispered count creates a relative dip, not absolute silence.

**Algorithm**:
```python
def find_loudness_dips(audio, sr, window_sec=1.0, dip_threshold_db=6.0):
    """Find windows that are significantly quieter than neighbors."""
    # Compute RMS for each 1-second window
    # Find windows where: db[i] < db[i-1] - threshold AND db[i] < db[i+1] - threshold
    # Return timestamps of dip centers
```

**Alternatives considered**:
- Shorter windows (0.5s) → More noise, less reliable
- Longer windows (2s) → Might miss short count-offs
- Adaptive threshold → Added complexity without clear benefit

### Decision 3: Use loudness dips to extend detected music boundaries backward

**Choice**: After AI detects music start, search backwards (up to 15 seconds) for a loudness dip. If found, extend the song boundary to start after that dip.

**Rationale**: This is robust to AI misclassification of intros. Even if the vocal beat is classified as speech, we'll catch it by extending backwards from the detected music. The dip marks where the conductor's count ends and the song truly begins.

**Search strategy**:
1. Find detected music start (e.g., 24:23)
2. Search backwards up to 15 seconds for loudness dips
3. If dip found (e.g., 24:19), extend song start to dip end (24:20)
4. If no dip found, keep original boundary

### Decision 4: Increase default minimum duration to 60 seconds

**Choice**: Change `--min-duration` default from 30s to 60s.

**Rationale**: User feedback indicates 30s captures interrupted attempts. Most complete choir songs are 2-4 minutes. 60s filters false positives while keeping all real songs.

**Alternatives considered**:
- Keep at 30s → Continues to produce false positives
- 90s or higher → Might miss short pieces or warm-up songs

## Risks / Trade-offs

**Risk: Humming in non-song contexts**
Adding "hum" to music scoring could cause false positives if someone hums during conversation between songs.
→ **Mitigation**: The smoothing and minimum duration filters should handle short humming. Monitor in testing.

**Risk: Loudness dip false positives**
Brief lulls in conversation could be detected as count-off dips.
→ **Mitigation**: Only use dips to extend backwards from already-detected music, not to detect new music regions. The dip must lead into continuous loud audio that becomes music.

**Risk: 60s minimum filters real short songs**
Some legitimate songs might be under 60 seconds.
→ **Mitigation**: The parameter is configurable via `--min-duration`. Document this for users with short pieces.
