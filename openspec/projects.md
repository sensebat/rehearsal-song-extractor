# Project Specification

## Requirements

### Version Control

- **Git**: Repository must be initialized with git
- All changes must be committed with clear, descriptive messages

### Package Management

- **uv**: Use `uv` for Python project and dependency management
- Project configuration in `pyproject.toml`
- No `requirements.txt` or `setup.py`

### Python Version

- **Python 3.10**: Pinned via `uv python pin 3.10`
- Required for TensorFlow/audio library compatibility on macOS
- Version constraint: `>=3.10,<3.11`

### System Dependencies

- **ffmpeg**: Required for audio extraction from video files
  - Install via Homebrew: `brew install ffmpeg`

## Project Structure

```
split-rehearsal-audio/
├── pyproject.toml          # Project config and dependencies
├── .python-version         # Python version pin (3.10)
├── .gitignore
├── README.md
├── extract_songs_ai.py     # AI-powered extraction (recommended)
├── extract_songs.py        # Heuristic-based extraction (fast)
├── openspec/
│   └── projects.md         # This file
├── temp/                   # Intermediate files (gitignored)
└── songs/                  # Output directory (gitignored)
```

## Dependencies

Core dependencies managed via `uv add`:

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch for ML inference |
| `torchaudio` | Audio processing for PyTorch |
| `transformers` | Hugging Face model loading |
| `accelerate` | Optimized model inference |
| `librosa` | Audio analysis and feature extraction |
| `pyannote.audio` | Audio processing utilities |

## Setup Commands

```bash
# Initialize project
git init
uv init --no-readme
uv python pin 3.10

# Install dependencies
uv add torch torchaudio transformers accelerate librosa pyannote.audio

# Install ffmpeg (macOS)
brew install ffmpeg
```

## Scripts

### extract_songs_ai.py (Primary)

AI-powered song extraction using MIT Audio Spectrogram Transformer.

- More accurate music/speech classification
- Leverages silence gaps for precise cut points
- Slower but robust

### extract_songs.py (Alternative)

Heuristic-based extraction using spectral features.

- Faster processing
- May confuse speech with music in some cases
- Good for quick previews
