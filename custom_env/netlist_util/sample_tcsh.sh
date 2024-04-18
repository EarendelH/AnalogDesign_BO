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
foreach directory ("$target_directory"/tmp*/)
  if (-d "$directory") then
    echo "Processing directory: $directory"

    # Process all .scs files in the directory
    foreach file ("$directory"/*.scs)
      if (-f "$file") then
        spectre -64 +aps "$file"
      endif
    end

    # Process all subdirectories
    foreach subdir ("$directory"/*/)
      if (-d "$subdir") then
        foreach file ("$subdir"/*)
          if (-f "$file") then
            psf "$file" -o "${file}.encode"
          endif
        end
      endif
    end
  endif
end

set end_time = `date +%s`
set elapsed_time = `expr $end_time - $start_time`

echo "Processing complete."
echo "Total time taken: $elapsed_time seconds."
