## Context

rse.py currently prints extraction progress and results to stdout only. After the terminal session ends, this information is lost. Users need to reference extraction details for debugging or reviewing what was extracted.

## Goals / Non-Goals

**Goals:**
- Capture all stdout output to a log file in the output directory
- Maintain existing console output behavior (user still sees real-time progress)

**Non-Goals:**
- Structured logging (JSON format, log levels, etc.)
- Log rotation or size limits
- Separate log streams for different types of messages

## Decisions

### Decision 1: Tee-style output

**Choice**: Use a custom print wrapper that writes to both stdout and file simultaneously.

**Rationale**: Simpler than redirecting stdout. Preserves real-time console output while capturing to file.

### Decision 2: Log file naming

**Choice**: `{label}.log` when label provided, otherwise `extraction.log` in the output directory.

**Rationale**: Matches the naming pattern already used for the output directory and files.

## Risks / Trade-offs

**Risk: Large log files**
For very long recordings, log files could grow large.
→ **Mitigation**: Not a concern for typical rehearsal recordings. Can add log rotation later if needed.
