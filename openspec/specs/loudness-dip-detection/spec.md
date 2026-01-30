## ADDED Requirements

### Requirement: Humming classified as music-related
The system SHALL include "humming" labels in the music score calculation when classifying audio chunks, treating humming as music-adjacent rather than speech.

#### Scenario: Vocal beat intro detected as music
- **WHEN** an audio chunk contains rhythmic vocal sounds (e.g., "wan wan wan" counting)
- **AND** the AI model classifies it primarily as "humming"
- **THEN** the chunk SHALL score higher for music than speech due to humming being counted as music-related

#### Scenario: Humming during conversation
- **WHEN** a brief humming sound occurs during speech/conversation
- **AND** the duration is less than the minimum music segment threshold
- **THEN** the humming SHALL be filtered out by the smoothing and duration filters

### Requirement: Loudness dip detection for song boundaries
The system SHALL detect song boundaries using coarse loudness analysis (1-second windows) to find count-off moments where audio volume drops significantly below neighboring windows.

#### Scenario: Count-off silence detected via loudness dip
- **WHEN** a 1-second window has RMS at least 6 dB below both its preceding and following windows
- **THEN** the system SHALL identify this as a potential song boundary marker

#### Scenario: Conductor whispered count detected
- **WHEN** a conductor whispers "1, 2, 3, and..." creating a relative quiet moment (not absolute silence)
- **AND** the RMS drops by at least 6 dB compared to surrounding chatter
- **THEN** the system SHALL detect this as a loudness dip even if frame-level silence detection finds no continuous silence

### Requirement: Backward boundary extension from detected music
The system SHALL search backwards from detected music start points to find loudness dips, extending the song boundary to capture vocal intros that were misclassified.

#### Scenario: Song boundary extended to count-off
- **WHEN** music is detected starting at time T
- **AND** a loudness dip exists within 15 seconds before T
- **THEN** the song boundary SHALL be extended backwards to the end of that loudness dip

#### Scenario: No dip found within search window
- **WHEN** music is detected starting at time T
- **AND** no loudness dip exists within 15 seconds before T
- **THEN** the song boundary SHALL remain at the original detected time T

#### Scenario: Vocal beat captured by extension
- **WHEN** a song starts with a vocal beat intro (e.g., rhythmic counting)
- **AND** the vocal beat is classified as speech/humming by the AI
- **AND** a count-off loudness dip precedes the vocal beat
- **THEN** the extended boundary SHALL include the entire vocal beat intro in the extracted song

### Requirement: Minimum song duration default
The system SHALL use 60 seconds as the default minimum song duration to filter interrupted attempts and false positives.

#### Scenario: Interrupted take filtered
- **WHEN** a music segment is detected with duration less than 60 seconds
- **AND** the user has not specified a custom --min-duration
- **THEN** the segment SHALL be excluded from the final song list

#### Scenario: Custom minimum duration honored
- **WHEN** the user specifies --min-duration 30
- **THEN** the system SHALL use 30 seconds as the minimum instead of the default 60 seconds
