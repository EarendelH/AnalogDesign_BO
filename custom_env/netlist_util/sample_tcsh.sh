#!/bin/tcsh

# Step 1: Get the target directory from user input
set target_dir = $1

# Check if the user has provided the target directory
if ("$target_dir" == "") then
    echo "Please provide a target directory."
    exit 1
endif

# Check if the target directory exists
if (! -d "$target_dir") then
    echo "Target directory does not exist."
    exit 1
endif

# Step 2: Find all subdirectories starting with tmp_
set tmp_dirs = (`find $target_dir -type d -name "tmp_*"`)

# Check if any tmp_ subdirectories were found
if ("$#tmp_dirs" == 0) then
    echo "No tmp_ directories found in the target directory."
    exit 1
endif

# Initialize lists to store run times and commands
set spectre_times = ()
set spectre_commands = ()
set psf_times = ()
set psf_commands = ()

# Step 3: Process each tmp_ subdirectory for .scs files
foreach tmp_dir ($tmp_dirs)
    set scs_files = (`find $tmp_dir -type f -name "*.scs"`)
    foreach scs_file ($scs_files)
        set start_time = `date +%s.%N`
        spectre -64 +aps $scs_file
        set end_time = `date +%s.%N`
        set duration = `echo "scale=2; $end_time - $start_time" | bc`
        set spectre_times = ($spectre_times $duration)
        set spectre_commands = ($spectre_commands "spectre -64 +aps $scs_file")
    end
end

# Step 4: Process .raw subdirectories for files
foreach tmp_dir ($tmp_dirs)
    set raw_dirs = (`find $tmp_dir -type d -name "*.raw"`)
    foreach raw_dir ($raw_dirs)
        set raw_files = (`find $raw_dir -type f`)
        foreach raw_file ($raw_files)
            set start_time = `date +%s.%N`
            psf $raw_file -o ${raw_file}.encode
            set end_time = `date +%s.%N`
            set duration = `echo "scale=2; $end_time - $start_time" | bc`
            set psf_times = ($psf_times $duration)
            set psf_commands = ($psf_commands "psf $raw_file -o ${raw_file}.encode")
        end
    end
end

# Step 5: Display all operation run times with commands
echo "Spectre command run times (seconds):"
@ i = 1
foreach time ($spectre_times)
    echo "$spectre_commands[$i] : $time"
    @ i++
end

echo "PSF command run times (seconds):"
@ i = 1
foreach time ($psf_times)
    echo "$psf_commands[$i] : $time"
    @ i++
end