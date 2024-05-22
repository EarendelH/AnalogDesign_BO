#!/bin/tcsh

set target_dir = $1

if ("$target_dir" == "") then
    echo "Please provide a target directory."
    exit 1
endif

if (! -d "$target_dir") then
    echo "Target directory does not exist."
    exit 1
endif

set tmp_dirs = (`find $target_dir -type d -name "tmp_*"`)

if ("$#tmp_dirs" == 0) then
    echo "No tmp_ directories found in the target directory."
    exit 1
endif

set spectre_times = ()
set psf_times = ()

foreach tmp_dir ($tmp_dirs)
    set scs_files = (`find $tmp_dir -type f -name "*.scs"`)
    foreach scs_file ($scs_files)
        set start_time = `date +%s`
        spectre -64 +aps $scs_file
        set end_time = `date +%s`
        @ duration = $end_time - $start_time
        set spectre_times = ($spectre_times $duration)
    end
end

foreach tmp_dir ($tmp_dirs)
    set raw_dirs = (`find $tmp_dir -type d -name "*.raw"`)
    foreach raw_dir ($raw_dirs)
        set raw_files = (`find $raw_dir -type f`)
        foreach raw_file ($raw_files)
            set start_time = `date +%s`
            psf $raw_file -o ${raw_file}.encode
            set end_time = `date +%s`
            @ duration = $end_time - $start_time
            set psf_times = ($psf_times $duration)
        end
    end
end

echo "Spectre command times (seconds):"
foreach time ($spectre_times)
    echo $time
end

echo "PSF command times (seconds):"
foreach time ($psf_times)
    echo $time
end