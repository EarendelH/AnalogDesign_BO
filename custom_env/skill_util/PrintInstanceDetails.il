;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure: PrintInstanceDetails
;;
;; Purpose: Prints detailed information about a specific instance in a cellview,
;;          including direct instance parameters and CDF parameters.
;;
;; Parameters:
;;   - libName: Library name
;;   - cellName: Cell name
;;   - viewName: View name
;;   - instName: Instance name to print details for
;;
;; Returns: None (prints information to console)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
procedure(PrintInstanceDetails(libName cellName viewName instName)
    let((cv inst cdfId)
        printf("Starting PrintInstanceDetails\n")

        ; Open the cellview in read mode
        cv = dbOpenCellViewByType(libName cellName viewName "" "r")
        printf("CellView opened: %L\n" cv)

        if(cv then
            ; Find the instance
            inst = dbFindAnyInstByName(cv instName)
            printf("Instance found: %L\n" inst)

            if(inst then
                printf("Details for instance %s:\n" instName)
                printf(" Cell Name: %s\n" inst~>cellName)
                printf(" Library Name: %s\n" inst~>libName)

                ; Print direct instance parameters
                printf("Direct Instance Parameters:\n")
                foreach(prop inst~>?
                    when(prop != 'name && prop != 'cellName && prop != 'libName
                        propValue = evalstring(sprintf(nil "inst~>%s" prop))
                        printf(" %s: %L\n" prop propValue)
                    )
                )

                ; Get and print CDF parameters
                cdfId = cdfGetInstCDF(inst)
                printf("CDF ID: %L\n" cdfId)
                when(cdfId
                    printf("CDF Parameters:\n")
                    foreach(param cdfId->parameters
                        paramValue = cdfFindParamByName(cdfId param->name)->value
                        printf(" %s: %L\n" param->name paramValue)

                        ; Print callback if it exists
                        when(param->callback
                            printf("  Callback: %s\n" param->callback)
                        )
                    )
                )
            else
                printf("Instance %s not found in %s/%s/%s\n" instName libName cellName viewName)
            )
            printf("Closing cellView\n")
            dbClose(cv)
        else
            printf("Failed to open cellView %s/%s/%s\n" libName cellName viewName)
        )
    )
)

; PrintInstanceDetails("LDO_Cai_Skill" "LDO_Core" "schematic" "M0")