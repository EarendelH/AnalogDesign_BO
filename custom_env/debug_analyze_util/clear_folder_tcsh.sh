#!/bin/tcsh


set dir = "/home/hanwu/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/run_test"

foreach folder (`find $dir -type d -name 'tmp_202405*'`)
    echo "Removing folder: $folder"
    rm -rf $folder
end