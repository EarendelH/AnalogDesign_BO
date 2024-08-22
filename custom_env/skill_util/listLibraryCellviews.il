procedure(listLibraryCellviews(libName)
    let((cells)
        ; Check if the library exists
        unless(ddGetObj(libName)
            printf("Error: Library '%s' not found.\n" libName)
            return()
        )

        ; Get all cells in the library
        cells = ddGetObj(libName)->cells

        if(cells then
            printf("Cellviews in library '%s':\n" libName)
            foreach(cell cells
                printf("  Cell: %s\n" cell~>name)
                foreach(view cell~>views
                    printf("    View: %s\n" view~>name)
                )
            )
            printf("Total cells: %d\n" length(cells))
        else
            printf("No cells found in library '%s'.\n" libName)
        )
    )
)

; Example usage:
; listLibraryCellviews("analogLib")