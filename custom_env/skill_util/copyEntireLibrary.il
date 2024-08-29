procedure(copyEntireLibrary(sourceLibName destinationLibName)
    let((srcSpec destSpec srcList destList copyResult updateLibList)
        printf("INFO: Starting to copy library from %s to %s\n" sourceLibName destinationLibName)

        ; Create source and destination library specs
        srcSpec = gdmCreateSpec(sourceLibName "" "" "" "CDBA")
        destSpec = gdmCreateSpec(destinationLibName "" "" "" "CDBA")
        unless(and(srcSpec destSpec)
            error("Failed to create specs for libraries")
        )

        ; Create spec lists
        srcList = gdmCreateSpecList()
        destList = gdmCreateSpecList()
        gdmAddSpecToSpecList(srcSpec srcList)
        gdmAddSpecToSpecList(destSpec destList)

        ; Create update library list (empty, but required)
        updateLibList = gdmCreateSpecList()

        ; Perform the copy operation
        printf("INFO: Starting copy operation...\n")
        copyResult = ccpCopy(srcList         ; Source library list
                             destList        ; Destination library list
                             t               ; Overwrite existing files
                             'CCP_EXPAND_ALL ; Expand and copy everything
                             nil             ; Copy all view types
                             nil             ; Copy all view names
                             ""              ; No name expression filter
                             "CDBA"          ; Use CDBA namespace
                             'CCP_UPDATE_DESTLIB_ONLY ; Update all cross-references in destination library
                             updateLibList   ; Empty update library list
                             nil             ; Don't copy property files
                             1               ; Validate layers, purposes, etc. after copy
                             1               ; Update custom via definitions
        )

        if(copyResult then
            printf("SUCCESS: Library %s has been successfully copied to %s\n" sourceLibName destinationLibName)
        else
            error("Failed to copy library %s to %s" sourceLibName destinationLibName)
        )

        copyResult
    )
)
