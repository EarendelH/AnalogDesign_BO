procedure(closeTechSaveDrmForm()
    let((form)
        form = hiGetForm("techSaveDrmForm")
        when(form
            hiFormUnmap(form)
            hiDeleteForm(form)
        )
    )
)