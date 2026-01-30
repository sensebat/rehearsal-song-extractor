## ADDED Requirements

### Requirement: Read YAML metadata file
The rst.py tool SHALL read a YAML metadata file and apply tags to MP3 files.

#### Scenario: Basic usage
- **WHEN** user runs `rst.py path/to/songs.yaml`
- **THEN** the tool SHALL read metadata from the YAML file
- **AND** apply tags to MP3 files in the same directory

### Requirement: Rename output directory
The tool SHALL rename the output directory based on artist and album metadata.

#### Scenario: Directory renamed
- **WHEN** YAML contains `artist: "Choir Name"` and `album: "2016-01-29 Generalprobe"`
- **THEN** the directory SHALL be renamed to `Choir Name - 2016-01-29 Generalprobe/`

### Requirement: Rename MP3 files with track titles
The tool SHALL rename MP3 files to `{{TRACK}} {{TITLE}}.mp3` format.

#### Scenario: Files renamed with titles
- **WHEN** YAML contains `tracks: ["Can you feel the love tonight", "Hallelujah"]`
- **THEN** `Song 01.mp3` SHALL be renamed to `01 Can you feel the love tonight.mp3`
- **AND** `Song 02.mp3` SHALL be renamed to `02 Hallelujah.mp3`

### Requirement: Apply ID3 tags to MP3 files
The tool SHALL write ID3v2 tags to each MP3 file.

#### Scenario: ID3 tags applied
- **WHEN** the tool processes an MP3 file
- **THEN** the following ID3 tags SHALL be set:
  - Title: track name from tracks list
  - Artist: from `artist` field
  - Album: from `album` field
  - Track number: position in tracks list (1, 2, 3...)
  - Year: from `year` field

### Requirement: Rename WAV files in Lossless directory
The tool SHALL also rename corresponding WAV files in the Lossless subdirectory.

#### Scenario: WAV files renamed
- **WHEN** MP3 `Song 01.mp3` is renamed to `01 Can you feel the love tonight.mp3`
- **THEN** `Lossless/Song 01.wav` SHALL be renamed to `Lossless/01 Can you feel the love tonight.wav`

### Requirement: Track count validation
The tool SHALL validate that track count matches file count.

#### Scenario: Count mismatch error
- **WHEN** YAML has 9 tracks but directory has 10 MP3 files
- **THEN** the tool SHALL exit with error
- **AND** display message indicating the mismatch
