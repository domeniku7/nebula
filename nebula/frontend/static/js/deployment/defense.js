// Defense Method Module
const DefenseMethodManager = {
    initialize() {
        this.setupSelector();
    },

    setupSelector() {
        const federationSelect = document.getElementById("federationArchitecture");
        const defenseDiv = document.getElementById("defense-method-div");
        const toggle = () => {
            if (!defenseDiv) return;
            defenseDiv.style.display = federationSelect.value === "DFL" ? "block" : "none";
        };
        if (federationSelect) {
            federationSelect.addEventListener("change", toggle);
            toggle();
        }
    },

    getDefenseMethod() {
        const select = document.getElementById("defenseMethodSelect");
        return select ? select.value : "none";
    },

    setDefenseMethod(method) {
        const select = document.getElementById("defenseMethodSelect");
        if (select) select.value = method || "none";
    },

    resetDefenseMethod() {
        this.setDefenseMethod("none");
    }
};

export default DefenseMethodManager;
