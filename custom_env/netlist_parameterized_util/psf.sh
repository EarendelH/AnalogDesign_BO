#!/bin/bash

echo "Enter the directory:"
read directory

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Directory does not exist."
  exit 1
fi

# Process all .scs files in the directory
for file in "$directory"/*.scs; do
  if [ -f "$file" ]; then
    spectre -64 +aps "$file"
  fi
done

# Process all subdirectories
for subdir in "$directory"/*/; do
  if [ -d "$subdir" ]; then
    for file in "$subdir"*; do
      if [ -f "$file" ]; then
        psf "$file" -o "${file}.encode"
      fi
    done
  fi
done

echo "Processing complete."