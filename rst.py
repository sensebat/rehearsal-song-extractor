#!/usr/bin/env python3
"""
Rehearsal Song Tagger - Apply metadata to extracted songs.

Reads a YAML metadata file and:
1. Renames the output directory to "Artist - Album"
2. Renames MP3/WAV files to "NN Title.ext"
3. Applies ID3 tags to MP3 files
"""

import argparse
import re
import sys
from pathlib import Path

import yaml
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TRCK, TDRC
from mutagen.mp3 import MP3


def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    # Replace problematic characters
    replacements = {
        "/": "-",
        "\\": "-",
        ":": "-",
        "*": "",
        "?": "",
        '"': "'",
        "<": "",
        ">": "",
        "|": "-",
    }
    for char, replacement in replacements.items():
        name = name.replace(char, replacement)
    # Remove leading/trailing whitespace and dots
    name = name.strip().strip(".")
    return name


def load_metadata(yaml_path: Path) -> dict:
    """Load and validate metadata from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required = ["artist", "album", "year", "tracks"]
    for field in required:
        if field not in data:
            print(f"Error: Missing required field '{field}' in {yaml_path}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(data["tracks"], list):
        print(f"Error: 'tracks' must be a list in {yaml_path}", file=sys.stderr)
        sys.exit(1)

    return data


def find_song_files(directory: Path) -> list[Path]:
    """Find MP3 files matching 'Song NN' pattern, sorted by number."""
    pattern = re.compile(r".*Song (\d+)\.mp3$", re.IGNORECASE)
    mp3_files = []

    for f in directory.glob("*.mp3"):
        match = pattern.match(f.name)
        if match:
            mp3_files.append((int(match.group(1)), f))

    # Sort by track number
    mp3_files.sort(key=lambda x: x[0])
    return [f for _, f in mp3_files]


def apply_id3_tags(mp3_path: Path, title: str, artist: str, album: str, track_num: int, year: int) -> None:
    """Apply ID3v2 tags to an MP3 file."""
    try:
        audio = MP3(mp3_path, ID3=ID3)
    except Exception:
        audio = MP3(mp3_path)

    # Create ID3 tag if it doesn't exist
    try:
        audio.add_tags()
    except Exception:
        pass  # Tags already exist

    audio.tags["TIT2"] = TIT2(encoding=3, text=title)
    audio.tags["TPE1"] = TPE1(encoding=3, text=artist)
    audio.tags["TALB"] = TALB(encoding=3, text=album)
    audio.tags["TRCK"] = TRCK(encoding=3, text=str(track_num))
    audio.tags["TDRC"] = TDRC(encoding=3, text=str(year))

    audio.save()


def main():
    parser = argparse.ArgumentParser(
        description="Apply metadata tags to extracted songs"
    )
    parser.add_argument(
        "yaml_file",
        type=Path,
        help="Path to songs.yaml metadata file",
    )

    args = parser.parse_args()

    if not args.yaml_file.exists():
        print(f"Error: File not found: {args.yaml_file}", file=sys.stderr)
        sys.exit(1)

    # Load metadata
    metadata = load_metadata(args.yaml_file)
    artist = metadata["artist"]
    album = metadata["album"]
    year = metadata["year"]
    tracks = metadata["tracks"]

    # Find MP3 files in same directory as YAML
    yaml_dir = args.yaml_file.parent
    mp3_files = find_song_files(yaml_dir)

    # Validate track count
    if len(tracks) != len(mp3_files):
        print(f"Error: Track count mismatch!", file=sys.stderr)
        print(f"  YAML has {len(tracks)} tracks", file=sys.stderr)
        print(f"  Directory has {len(mp3_files)} MP3 files", file=sys.stderr)
        print(f"\nPlease ensure the track list matches the extracted files.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(tracks)} songs...")
    print(f"  Artist: {artist}")
    print(f"  Album: {album}")
    print(f"  Year: {year}")

    # Rename files and apply tags
    lossless_dir = yaml_dir / "Lossless"

    for i, (mp3_path, title) in enumerate(zip(mp3_files, tracks), 1):
        safe_title = sanitize_filename(title)
        new_name = f"{i:02d} {safe_title}"

        # Rename MP3
        new_mp3_path = yaml_dir / f"{new_name}.mp3"
        mp3_path.rename(new_mp3_path)

        # Rename corresponding WAV if it exists
        old_wav_name = mp3_path.stem + ".wav"
        old_wav_path = lossless_dir / old_wav_name
        if old_wav_path.exists():
            new_wav_path = lossless_dir / f"{new_name}.wav"
            old_wav_path.rename(new_wav_path)

        # Apply ID3 tags
        apply_id3_tags(new_mp3_path, title, artist, album, i, year)

        print(f"  {i:02d}. {title}")

    # Rename directory to "Artist - Album"
    if artist and album:
        new_dir_name = sanitize_filename(f"{artist} - {album}")
        new_dir_path = yaml_dir.parent / new_dir_name

        if new_dir_path != yaml_dir:
            if new_dir_path.exists():
                print(f"\nWarning: Cannot rename directory - '{new_dir_name}' already exists")
            else:
                yaml_dir.rename(new_dir_path)
                print(f"\nRenamed directory to: {new_dir_name}/")

    print("\nDone! Songs tagged and renamed.")


if __name__ == "__main__":
    main()
