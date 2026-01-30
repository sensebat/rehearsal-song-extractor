## ADDED Requirements

### Requirement: Optional label argument for output naming
The system SHALL accept an optional `--label` argument that specifies a prefix for all output files and the output directory name.

#### Scenario: Label provided
- **WHEN** user runs `rse.py input.mov --label "2016-01-29 Generalprobe"`
- **THEN** output directory SHALL be named `2016-01-29 Generalprobe/`
- **AND** all output files SHALL be prefixed with `2016-01-29 Generalprobe `

#### Scenario: Label omitted
- **WHEN** user runs `rse.py input.mov` without `--label`
- **THEN** output directory SHALL be named `songs/`
- **AND** output files SHALL have no prefix (e.g., `Song 01.mp3`, `Song 01.wav`)

#### Scenario: Label with special characters
- **WHEN** user provides a label containing spaces or hyphens
- **THEN** the label SHALL be used exactly as provided in directory and file names

### Requirement: Structured output directory
The system SHALL create a structured output directory with MP3 files at root and WAV files in a Lossless subdirectory.

#### Scenario: Output structure created
- **WHEN** songs are exported with label "MyLabel"
- **THEN** the output structure SHALL be:
  ```
  MyLabel/
  |   Lossless/
  |   |   MyLabel Song 01.wav
  |   |   MyLabel Song 02.wav
  |   MyLabel Song 01.mp3
  |   MyLabel Song 02.mp3
  ```

### Requirement: MP3 encoding at 320 kbps
The system SHALL encode extracted songs to MP3 format at 320 kbps bitrate.

#### Scenario: MP3 files created
- **WHEN** a song is exported
- **THEN** an MP3 file SHALL be created at 320 kbps constant bitrate
- **AND** the MP3 SHALL be placed in the root of the label directory

### Requirement: Lossless WAV preservation
The system SHALL preserve the original lossless WAV files in a Lossless subdirectory.

#### Scenario: WAV files preserved
- **WHEN** a song is exported
- **THEN** the WAV file SHALL be saved in the `Lossless/` subdirectory
- **AND** the WAV SHALL retain original sample rate and bit depth from source

### Requirement: Song numbering format
The system SHALL number songs with zero-padded two-digit format.

#### Scenario: Song numbering
- **WHEN** songs are exported
- **THEN** files SHALL be named `{{LABEL}} Song 01`, `{{LABEL}} Song 02`, etc.
