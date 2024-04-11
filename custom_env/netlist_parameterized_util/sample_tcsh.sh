#!/bin/tcsh

echo "Enter the directory:"
set directory = $<
set start_time = `date +%s`

# Check if the directory exists
if (! -d "$directory") then
  echo "Directory does not exist."
  exit 1
endif

# Process all .scs files in the directory
foreach file ("$directory"/*.scs)
  if (-f "$file") then
    spectre -64 +aps "$file"
  endif
end

# Process all subdirectories
foreach subdir ("$directory"/*/)
  if (-d "$subdir") then
    foreach file ("$subdir"*)
      if (-f "$file") then
        psf "$file" -o "${file}.encode"
      endif
    end
  endif
end

set end_time = `date +%s`
set elapsed_time = `expr $end_time - $start_time`

echo "Processing complete."
echo "Total time taken: $elapsed_time seconds."
