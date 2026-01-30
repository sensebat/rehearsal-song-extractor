## ADDED Requirements

### Requirement: YAML template generation after extraction
The system SHALL generate a YAML metadata template file after extracting songs.

#### Scenario: Template created with label
- **WHEN** rse.py extracts songs with `--label "2016-01-29 Generalprobe"`
- **THEN** a file `2016-01-29 Generalprobe/songs.yaml` SHALL be created
- **AND** the `album` field SHALL be pre-filled with the label

#### Scenario: Template created without label
- **WHEN** rse.py extracts songs without `--label`
- **THEN** a file `songs/songs.yaml` SHALL be created
- **AND** the `album` field SHALL be empty

### Requirement: YAML template structure
The YAML template SHALL contain the following fields.

#### Scenario: Template field structure
- **WHEN** the YAML template is generated
- **THEN** it SHALL contain:
  ```yaml
  artist: ""
  album: "{{LABEL or empty}}"
  year: {{CURRENT_YEAR}}
  tracks:
    - "Song 01"
    - "Song 02"
    # ... one entry per extracted song
  ```

### Requirement: Track list matches extracted files
The tracks list SHALL have one entry per extracted MP3 file.

#### Scenario: Track list populated
- **WHEN** 9 songs are extracted
- **THEN** the tracks list SHALL have 9 entries
- **AND** entries SHALL be placeholder names matching file names (e.g., "Song 01", "Song 02")
