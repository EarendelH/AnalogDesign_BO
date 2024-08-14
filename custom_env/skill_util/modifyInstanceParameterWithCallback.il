;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure: modifyInstanceParameterWithCallback
;;
;; Purpose: Modifies a parameter of an instance in a cellview, updates the CDF,
;;          calls the associated callback, and saves the changes.
;;
;; Parameters:
;;   - libName: Library name
;;   - cellName: Cell name
;;   - viewName: View name
;;   - instName: Instance name
;;   - paramName: Parameter name to modify
;;   - newValue: New value for the parameter
;;
;; Returns: t if the parameter was changed, nil otherwise
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
procedure(modifyInstanceParameterWithCallback(libName cellName viewName instName paramName newValue)
    let((cv inst changed cdfId oldValue)
        ; Print start message
        printf("Starting modifyInstanceParameterWithCallback\n")

        ; Open the cellview
        cv = dbOpenCellViewByType(libName cellName viewName "" "a")
        printf("CellView opened: %L\n" cv)

        if(cv then
            ; Find the instance
            inst = dbFindAnyInstByName(cv instName)
            printf("%s instance found: %L\n" instName inst)

            if(inst then
                ; Get the old value of the parameter
                oldValue = evalstring(strcat("inst~>" paramName))
                printf("Old %s value: %L\n" paramName oldValue)
                printf("New %s value to set: %L\n" paramName newValue)

                ; Get the CDF (Component Description Format) ID
                cdfId = cdfGetInstCDF(inst)
                if(cdfId then
                    ; Find the parameter in the CDF
                    param = cdfFindParamByName(cdfId paramName)
                    if(param then
                        ; Update the CDF parameter value
                        param->value = newValue
                        printf("CDF %s updated: %L\n" paramName param->value)

                        ; Call the callback function if it exists
                        when(param~>callback
                            printf("Calling callback for %s\n" paramName)
                            errset(evalstring(param~>callback))
                        )

                        ; Update instance parameters based on CDF values
                        cdfUpdateInstParam(inst)
                        printf("Instance parameters updated from CDF\n")

                        ; Print all updated parameter values
                        foreach(param cdfId->parameters
                            updatedValue = evalstring(strcat("inst~>" param->name))
                            printf("Updated %s: %L\n" param->name updatedValue)
                        )

                        changed = t
                    else
                        printf("Parameter %s not found in CDF\n" paramName)
                    )

                    ; Save changes to the cellview
                    dbSave(cv)
                    printf("CellView saved\n")
                else
                    printf("CDF data not found for instance %s\n" instName)
                )
            else
                printf("%s instance not found in %s/%s/%s\n" instName libName cellName viewName)
            )
            printf("Closing cellView\n")
            dbClose(cv)
        else
            printf("Failed to open cellView %s/%s/%s\n" libName cellName viewName)
        )
        changed
    )
)