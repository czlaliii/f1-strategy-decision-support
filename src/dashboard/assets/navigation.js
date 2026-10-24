// Tab navigation of the replay workspace, handled in the browser.
// Returns the panel styles followed by the tab button classes.
(function () {
    var TABS = ["track", "strategy", "laps", "progress"];

    function switchTab() {
        var context = window.dash_clientside.callback_context;
        var active = "track";
        if (context && context.triggered && context.triggered.length && context.triggered[0].prop_id !== ".") {
            active = context.triggered[0].prop_id.split(".")[0].replace("nav-", "");
        }
        // Plotly graphs drawn while their panel was hidden need a resize to fill it.
        window.setTimeout(function () {
            window.dispatchEvent(new Event("resize"));
        }, 60);
        var styles = [];
        var classes = [];
        for (var i = 0; i < TABS.length; i++) {
            styles.push(TABS[i] === active ? {} : {display: "none"});
            classes.push(TABS[i] === active ? "nav-tab active" : "nav-tab");
        }
        return styles.concat(classes);
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        nav: {switchTab: switchTab}
    });
})();
