#!/bin/bash

INPUT_DIR="duet_svs/Choir_Dataset"

find "$INPUT_DIR" -type f -name "*.mp3" | while read -r mp3_file; do
    wav_file="${mp3_file%.mp3}.wav"
    echo "Converting: $mp3_file → $wav_file"
    ffmpeg -y -i "$mp3_file" "$wav_file"
done
