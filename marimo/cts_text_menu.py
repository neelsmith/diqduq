import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Browse citable texts
    """)
    return


@app.cell(hide_code=True)
def _(uploaded_file):
    uploaded_file
    return


@app.cell
def _(selected_path):
    selected_path
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Implementation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Imports
    """)
    return


@app.cell
def _():
    from pathlib import Path

    return (Path,)


@app.cell
def _(uploaded_file):
    uploaded_file.value[0]
    return


@app.cell
def _(records):
    records
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## UI
    """)
    return


@app.cell
def _(mo):
    uploaded_file = mo.ui.file(label="Choose a CTS text file")
    return (uploaded_file,)


@app.cell
def _(uploaded_file):
    str_raw = uploaded_file.value[0].contents.decode()
    return (str_raw,)


@app.cell
def _(str_raw):
    lines = str_raw.split("\n")[2:]
    return (lines,)


@app.cell
def _(lines):
    u,t = lines[0].split("|")
    return


@app.cell
def _(lines):
    prs = [ln.split("|") for ln in lines]
    return (prs,)


@app.cell
def _(prs):
    prs
    return


@app.cell
def _(uploaded_file):
    selected_path = resolve_uploaded_path(uploaded_file.value)
    return (selected_path,)


@app.function
def resolve_uploaded_path(upload):
    if not upload:
        return None

    if isinstance(upload, list):
        upload = upload[0]

    if isinstance(upload, dict):
        path = upload.get("path") or upload.get("name")
    elif hasattr(upload, "path"):
        path = upload.path
    else:
        path = str(upload)

    return path


@app.cell
def _():
    #if not selected_path:
    #    mo.md("No file selected yet.")
    #else:
    #    mo.md(f"Loaded file: `{selected_path}`")
    return


@app.cell
def _(mo, records, selected_path):
    report_rec_count = None
    if not selected_path:
       report_rec_count =  mo.md("No file selected yet.")
    elif not records:
       report_rec_count =  mo.md(f"Selected file: `{selected_path}`\n\nNo valid CTS records were found.")
    else:
        report_rec_count = mo.md(f"Selected file: `{selected_path}`\n\nParsed {len(records)} passages.")

    report_rec_count
    return


@app.cell
def _(parse_cts_file, selected_path):
    records = parse_cts_file(selected_path) if selected_path else []
    return (records,)


@app.cell
def _(Path):

    def parse_cts_file(path_str):
        if not path_str:
            return []

        path = Path(path_str)
        if not path.exists():
            return []

        records = []
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue

                if "|" in line:
                    delim = "|"
                elif "\t" in line:
                    delim = "\t"
                else:
                    continue

                parts = [part.strip() for part in line.split(delim, 1)]
                if len(parts) != 2:
                    continue

                urn, text = parts
                if not urn or not text:
                    continue

                ref = urn.rsplit(":", 1)[-1]
                words = text.split()
                preview = " ".join(words[:3])
                records.append(
                    {
                        "urn": urn,
                        "ref": ref,
                        "text": text,
                        "preview": preview,
                    }
                )

        return records



    return (parse_cts_file,)


@app.cell
def _(mo, records):
    passage_menu = None
    if records:

        menu_options = [f"{record['ref']} — {record['preview']}" for record in records]

        passage_menu = mo.ui.dropdown(
            label="Select a passage",
            options=menu_options,
            value=menu_options[0],
        )
    passage_menu    
    return


app._unparsable_cell(
    r"""
    if not records or passage_menu is None:
        mo.md("Choose a valid CTS text file to view passage text.")
        return

    selected_label = passage_menu.value
    selected = next(
        (
            record
            for record in records
            if f"{record['ref']} — {record['preview']}" == selected_label
        ),
        records[0],
    )

    mo.md(
        f"## {selected['ref']}\n\n"
        f"**URN:** {selected['urn']}\n\n"
        f"{selected['text']}"
    )
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
