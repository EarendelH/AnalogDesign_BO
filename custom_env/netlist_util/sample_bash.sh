#!/bin/bash

echo "Enter the target directory:"
read target_directory

start_time=$(date +%s)

# Check if the target directory exists
if [ ! -d "$target_directory" ]; then
  echo "Target directory does not exist."
  exit 1
fi

# Loop through all subdirectories starting with 'tmp'
for directory in "$target_directory"/tmp*/; do
  if [ -d "$directory" ]; then
    echo "Processing directory: $directory"

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
  fi
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo "Processing complete."
echo "Total time taken: $elapsed_time seconds."
