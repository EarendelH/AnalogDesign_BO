;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure: showCellViewInstances
;;
;; Purpose: Displays all instances in a given cellview
;;
;; Parameters:
;;   - libName: Library name
;;   - cellName: Cell name
;;   - viewName: View name
;;
;; Returns: None (prints information to console)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
procedure(showCellViewInstances(libName cellName viewName)
    let((cv instances)
        ; Open the cellview
        cv = dbOpenCellViewByType(libName cellName viewName "" "r")

        if(cv then
            printf("Displaying instances in %s/%s/%s:\n" libName cellName viewName)

            ; Get all instances in the cellview
            instances = cv~>instances

            if(instances then
                foreach(inst instances
                    printf("Instance: %s\n" inst~>name)
                    printf("  Cell: %s\n" inst~>cellName)
                    printf("  Library: %s\n" inst~>libName)
                    printf("  Master: %L\n" inst~>master)
                    printf("  Bounding Box: %L\n" inst~>bBox)
                    printf("\n")
                )
                printf("Total instances: %d\n" length(instances))
            else
                printf("No instances found in this cellview.\n")
            )

            ; Close the cellview
            dbClose(cv)
        else
            printf("Failed to open cellview %s/%s/%s\n" libName cellName viewName)
        )
    )
)

; Example usage:
; showCellViewInstances("LDO_Cai" "LDO" "schematic")