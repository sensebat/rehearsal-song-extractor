## ADDED Requirements

### Requirement: Log file creation
The system SHALL create a log file in the output directory when extracting songs.

#### Scenario: Log file with label
- **WHEN** user runs `rse.py audio.mov --label "2016-01-29 Generalprobe"`
- **THEN** the system SHALL create `2016-01-29 Generalprobe/2016-01-29 Generalprobe.log`

#### Scenario: Log file without label
- **WHEN** user runs `rse.py audio.mov` (no label)
- **THEN** the system SHALL create `songs/extraction.log`

### Requirement: Log file content
The log file SHALL contain the same output that is printed to the console.

#### Scenario: Console output captured
- **WHEN** extraction completes
- **THEN** the log file SHALL contain all messages that were printed to stdout during extraction

### Requirement: Real-time console output preserved
The system SHALL continue to print output to the console in real-time.

#### Scenario: Dual output
- **WHEN** extraction is in progress
- **THEN** the user SHALL see real-time progress in the terminal
- **AND** the same output SHALL be written to the log file
