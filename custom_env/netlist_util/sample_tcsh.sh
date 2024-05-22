#!/bin/tcsh

echo "Enter the target directory:"
set target_directory = $<

set start_time = `date +%s`

# Check if the target directory exists
if (! -d "$target_directory") then
  echo "Target directory does not exist."
  exit 1
endif

# Loop through all subdirectories starting with 'tmp'
foreach directory (`ls -d $target_directory/tmp*/`)
  if (-d "$directory") then
    echo "Processing directory: $directory"

    # Process all .scs files in the directory
    foreach file (`ls $directory/*.scs 2>/dev/null`)
      if (-f "$file") then
        set start_spectre = `date +%s`
        spectre -64 +aps "$file"
        set end_spectre = `date +%s`
        @ elapsed_spectre = $end_spectre - $start_spectre
        echo "Spectre processing time for $file: $elapsed_spectre seconds."
      endif
    end

    # Process all subdirectories
    foreach subdir (`ls -d $directory/*/ 2>/dev/null`)
      if (-d "$subdir") then
        foreach file (`ls $subdir/* 2>/dev/null`)
          if (-f "$file") then
            set start_psf = `date +%s`
            psf "$file" -o "${file}.encode"
            set end_psf = `date +%s`
            @ elapsed_psf = $end_psf - $start_psf
            echo "PSF processing time for $file: $elapsed_psf seconds."
          endif
        end
      endif
    end
  endif
end

set end_time = `date +%s`
@ elapsed_time = $end_time - $start_time

echo "Processing complete."
echo "Total time taken: $elapsed_time seconds."