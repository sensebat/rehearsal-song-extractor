## 1. Logging Infrastructure

- [x] 1.1 Add TeeLogger class that writes to both stdout and file
- [x] 1.2 Add log file path logic: `{label}.log` or `extraction.log`

## 2. Integration

- [x] 2.1 Create log file at start of extraction (after output dir created)
- [x] 2.2 Replace print() calls with logger or redirect stdout

## 3. Testing

- [x] 3.1 Run extraction and verify log file created with correct name
- [x] 3.2 Verify log file contains same output as console
