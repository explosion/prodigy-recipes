function checkBigModel() {
    document.querySelector("#loadingIcon").style.display = "inline-block"
    window.prodigy.event("check_big_model", { task: window.prodigy.content })
        .then(res => {
            window.prodigy.update(res)
            document.querySelector("#loadingIcon").style.display = "none"
        }).catch(err => {
            console.error("ERROR", err)
        })
    }
