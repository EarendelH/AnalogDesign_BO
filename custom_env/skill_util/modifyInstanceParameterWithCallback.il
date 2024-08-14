procedure(ModifyInstanceParameter(libName cellName viewName instName paramName newValue)
    printf("Modifying %s for instance %s in %s/%s/%s\n" paramName instName libName cellName viewName)
    let((cv inst cdfgData paramType oldValue updatedValue)
        cv = dbOpenCellViewByType(libName cellName viewName "" "a")
        if(cv then
            inst = dbFindAnyInstByName(cv instName)
            if(inst then
                cdfgData = cdfGetInstCDF(inst)
                if(cdfgData then
                    paramType = cdfFindParamByName(cdfgData paramName)
                    if(paramType then
                        oldValue = paramType->value
                        printf("Old %s value: %L\n" paramName oldValue)
                        printf("New %s value to set: %L\n" paramName newValue)

                        ; Update CDF parameter
                        paramType->value = newValue
                        printf("CDF %s updated: %L\n" paramName paramType->value)

                        ; Execute callback if exists
                        if(paramType~>callback then
                            printf("Executing callback for '%s' parameter\n" paramName)
                            evalstring(paramType~>callback)
                        )

                        ; Update instance parameters based on CDF values
                        cdfUpdateInstParam(inst)
                        printf("Instance parameters updated from CDF\n")

                        ; Print updated value
                        updatedValue = cdfFindParamByName(cdfgData paramName)->value
                        printf("Updated %s: %L\n" paramName updatedValue)

                        ; Save changes
                        dbSave(cv)
                        printf("CellView saved\n")
                    else
                        printf("Parameter %s not found for instance %s\n" paramName instName)
                    )
                else
                    printf("CDF data not found for instance %s\n" instName)
                )
            else
                printf("Instance %s not found in the cellview\n" instName)
            )
            dbClose(cv)
        else
            printf("Failed to open cellview %s/%s/%s\n" libName cellName viewName)
        )
    )
)

; Example usage
; ModifyInstanceParameter("LDO_Cai_Skill" "LDO_Core" "schematic" "M0" "w" "25u")
; ModifyInstanceParameter("LDO_Cai_Skill" "LDO_Core" "schematic" "M0" "l" "1u")
; ModifyInstanceParameter("LDO_Cai_Skill" "LDO_Core" "schematic" "M0" "fingers" "10")