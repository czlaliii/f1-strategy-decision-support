from src.dashboard.app import create_app


def _layout_ids(component):
    ids = set()
    if getattr(component, "id", None) is not None:
        ids.add(component.id)
    children = getattr(component, "children", None)
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            ids |= _layout_ids(child)
    return ids


def test_every_callback_id_exists_in_layout():
    app = create_app()
    layout_ids = _layout_ids(app.layout)
    missing = set()
    for output_key, callback in app.callback_map.items():
        # Multi-output keys look like "..id.prop...id2.prop2.."; "@hash" marks allow_duplicate.
        for output in output_key.strip(".").split("..."):
            missing.add(output.split("@")[0].rsplit(".", 1)[0])
        for dependency in callback["inputs"] + callback["state"]:
            missing.add(dependency["id"])
    assert missing - layout_ids == set()


def test_no_component_starts_hidden_without_a_callback_to_show_it():
    app = create_app()
    shown_by_callback = {
        output.split("@")[0].rsplit(".", 1)[0]
        for key in app.callback_map
        for output in key.strip(".").split("...")
        if output.split("@")[0].endswith(".style")
    }

    def hidden(component):
        found = set()
        if (getattr(component, "style", None) or {}).get("display") == "none":
            found.add(getattr(component, "id", None))
        children = getattr(component, "children", None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if hasattr(child, "to_plotly_json"):
                found |= hidden(child)
        return found

    assert hidden(app.layout) - shown_by_callback == set()
