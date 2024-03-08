#!/bin/bash

echo "Process Folder Name"
read folder_name

if [ ! -d "$folder_name" ]; then
  echo "No such folder"
  exit 1
fi

for file in "$folder_name"/*; do
  if [ -f "$file" ]; then
    psf "$file" -o "${file}.encode"
  fi

done
