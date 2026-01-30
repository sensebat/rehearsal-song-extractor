## Why

When rse.py extracts songs, the console output shows useful information (detected songs, timestamps, confidence scores) that is lost after the terminal session ends. Storing this output in a log file makes it available for later review and debugging.

## What Changes

- Write all console output to a `.log` file in the output directory
- Log file named `{label}.log` when label provided, otherwise `extraction.log`
- Existing console output behavior unchanged (still prints to stdout)

## Capabilities

### New Capabilities
- `extraction-log`: Store rse.py console output in a log file alongside extracted songs

### Modified Capabilities
<!-- None - this is purely additive -->

## Impact

- `rse.py`: Add tee-style logging to capture stdout to file
- Output directory: Will contain new `.log` file after extraction
