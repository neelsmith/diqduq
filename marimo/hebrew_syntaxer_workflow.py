import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo


    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Analyze Biblical Hebrew syntax with a configured LM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    *Enter values for a base URN (context), a passage reference, and text to analyze, then submit the form with the `Analyze` button.*
    """)
    return


@app.cell(hide_code=True)
def _(input_form):
    input_form
    return


@app.cell(hide_code=True)
def _(mo, results):
    mo.md("**Discussion**:\n\n" + "\n\n".join(f"> {result.reasoning}" for result in results))
    return


@app.cell(hide_code=True)
def _(psghtml):
    psghtml
    return


@app.cell(hide_code=True)
def _(vuhtml):
    vuhtml
    return


@app.cell(hide_code=True)
def _(indentpsg):
    indentpsg
    return


@app.cell(hide_code=True)
def _(diagram, mo):
    mo.mermaid(diagram)
    return


@app.cell(hide_code=True)
def _(analysis_warnings, download_widget, mo, save_extension):
    mo.vstack(
        [
            mo.hstack([save_extension, download_widget], justify="start"),
        ]
        + (
            [mo.callout(mo.md("\n".join(f"- {w}" for w in analysis_warnings)), kind="warn")]
            if analysis_warnings
            else []
        )
    )
    return


@app.cell(hide_code=True)
def _(download_mermaid):
    download_mermaid
    return


@app.cell(hide_code=True)
def _(mo):
    seetokens = mo.ui.checkbox(label="*See list of tokens*")
    seecost = mo.ui.checkbox(label="*See cost*")
    seeprompts = mo.ui.checkbox(label="*See prompts*")
    mo.hstack([seetokens, seeprompts, seecost], justify="start")
    return seecost, seeprompts, seetokens


@app.cell(hide_code=True)
def _(finaltokens, seetokens):
    tokendisplay = None
    if seetokens.value:
        tokendisplay = finaltokens

    tokendisplay
    return


@app.cell(hide_code=True)
def _(cost, mo, seecost):
    costdisplay = None
    if seecost.value:
        costdisplay = mo.md(f"**Total cost**: {cost}")
    costdisplay
    return


@app.cell(hide_code=True)
def _(dspy, seeprompts):
    prompts = None
    if seeprompts.value:
        prompts = dspy.inspect_history()
    prompts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("<hr/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>")
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
    ## Analysis
    """)
    return


@app.cell
def _(analyze_passage, input_form):
    # Analyze text passage -- only once the form has been submitted at
    # least once (input_form.value is None until then), and only again on
    # each subsequent submission, not on every keystroke in the form's own
    # inputs.
    passage = ''
    sentences, results = [], []
    if input_form.value and input_form.value.get("text_area"):
        passage = input_form.value["text_area"]
        citation = input_form.value["urnbase"] + input_form.value["citation_context"]
        sentences, results = analyze_passage(passage, citation=citation)
    return results, sentences


@app.cell
def _(combined_tokengraph, results, tokengraph_to_mermaid):
    # Compose Mermaid diagram: tokengraph_to_mermaid() is diqduq's own
    # utility for turning a tokengraph into a Mermaid flowchart string
    # (diqduq/mermaid.py), one node per substantive token and one edge per
    # relation, colored by verbal unit -- see that module's own docstring.
    finaltokens = combined_tokengraph(results)
    diagram, mermaid_warnings = tokengraph_to_mermaid(finaltokens)
    return diagram, finaltokens


@app.cell
def _(sentences):
    tokens = [tok for sentence in sentences for tok in sentence.tokens]
    return


@app.cell
def _(results):
    vus = [res.verbalunits for res in results]
    return


@app.cell
def _(last_call):
    # last_call is None until the form has been submitted at least once (no
    # LM calls yet, so lm.history is empty) -- guard against that rather
    # than assuming a call has already happened, unlike arsgrammatica's own
    # analogous cell, which doesn't guard this and fails the same way this
    # notebook did before this fix, on a fresh, not-yet-submitted load.
    cost = last_call.get('cost') if last_call is not None else None
    return (cost,)


@app.cell
def _(lm):
    last_call = None
    if lm.history:
        last_call = lm.history[-1]
    return (last_call,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Format output display
    """)
    return


@app.cell
def _(finaltokens, input_form, mo, tokengraph_to_text):
    citation_label = input_form.value["citation_context"] if input_form.value else ""
    psghtml = mo.Html(f"<b><i>Passage {citation_label}</i></b>: " + tokengraph_to_text(finaltokens))
    return (psghtml,)


@app.cell
def _(finaltokens, mo, tokengraph_to_html):
    # tokengraph_to_html()'s own include_cantillation=False: diqduq's HTML
    # utility for a verbal-unit-colored reading view (diqduq/rendering.py),
    # here asked to drop cantillation marks (te'amim, e.g. the verse-final
    # sof pasuq) entirely rather than just leaving them uncolored, so this
    # view reads as plain highlighted prose without accent marks mixed in.
    vuhtml = mo.Html(
        "<b><i>Highlighted by verbal unit</i></b>: "
        + tokengraph_to_html(finaltokens, include_cantillation=False)
    )
    return (vuhtml,)


@app.cell
def _(finaltokens, mo, tokengraph_to_depth_html):
    indenthtml, indentwarnings = tokengraph_to_depth_html(finaltokens)
    indentpsg = mo.Html("<b><i>Indented by verbal unit</i></b>: " + indenthtml)
    return (indentpsg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Save analysis
    """)
    return


@app.cell
def _(finaltokens, results, sentences, serialize_analyses):
    # Flatten every sentence's own verbalunits into the one flat list
    # serialize_analyses()/write_analyses() expect, matching how
    # combined_tokengraph() already flattens tokengraph across sentences.
    all_verbalunits = [vu for result in results for vu in result.verbalunits]
    analysis_text, analysis_warnings = serialize_analyses(sentences, all_verbalunits, finaltokens)
    return analysis_text, analysis_warnings


@app.cell
def _(input_form):
    # A readable default filename base, drawn from whatever citation the
    # form was submitted with (falling back to "analysis" if the passage
    # field was left blank) -- the extension is chosen separately, via
    # save_extension below.
    filename_base = ""
    if input_form.value:
        filename_base = (input_form.value.get("urnbase") or "") + (input_form.value.get("citation_context") or "")
    filename_base = "".join(c if c.isalnum() else "_" for c in filename_base).strip("_") or "analysis"
    return (filename_base,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## UI
    """)
    return


@app.cell
def _(mo):
    urnbase = mo.ui.text(
        value="urn:cts:compnov:bible.genesis.masoretic:",
        label="*Base URN (context)*:",
    )
    return (urnbase,)


@app.cell
def _(mo):
    citation_context = mo.ui.text(placeholder="1.1", label="*Passage reference*:")
    return (citation_context,)


@app.cell
def _(mo):
    text_area = mo.ui.text_area(
        value="בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃",
        full_width=True,
        label="*Text to analyze*:",
    )
    return (text_area,)


@app.cell
def _(citation_context, mo, text_area, urnbase):
    # All three inputs as one form -- marimo only updates input_form.value
    # (and so only re-triggers the Analysis cell below) when the whole form
    # is submitted, never on every keystroke in an individual field.
    input_form = (
        mo.md(
            """
            {urnbase}

            {citation_context}

            {text_area}
            """
        )
        .batch(urnbase=urnbase, citation_context=citation_context, text_area=text_area)
        .form(submit_button_label="Analyze")
    )
    return (input_form,)


@app.cell
def _(mo):
    save_extension = mo.ui.radio(
        options=["cex", "txt"], value="cex", inline=True, label="*File extension*:"
    )
    return (save_extension,)


@app.cell
def _(analysis_text, filename_base, mo, results, save_extension):
    # mo.download() puts the browser in charge of where the file lands --
    # no folder-path field to mistype, at the cost of not choosing a
    # location up front (the browser's own download prompt/default
    # download folder decides that). filename reactively follows both the
    # citation-derived filename_base and whichever extension is chosen
    # above.
    download_widget = mo.download(
        data=analysis_text.encode("utf-8"),
        filename=f"{filename_base}.{save_extension.value}",
        label="Download analysis",
        mimetype="text/plain",
        disabled=not results,
    )
    return (download_widget,)


@app.cell
def _(diagram, filename_base, mo):
    download_mermaid = mo.download(
        data=("```mermaid\n\n" + diagram + "\n```\n").encode("utf-8"),
        filename=f"{filename_base}.md",
        label="Download mermaid diagram",
        mimetype="text/plain",
    )
    return (download_mermaid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Imports
    """)
    return


@app.cell
def _():
    import dspy

    return (dspy,)


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Reuse diqduq_main.py's own .env-loading + LM-config helper
    # (_configure_lm, aliased here since marimo treats an
    # underscore-prefixed name as private to the cell that defines it)
    # rather than duplicating its MODEL/API_KEY handling here, the way
    # arsgrammatica's own marimo notebooks each build their own ad hoc
    # configure_lm(). diqduq_main.py's own top-level load_dotenv() call
    # already runs on this import, resolved relative to diqduq_main.py's
    # own location (the repo root) regardless of where this notebook file
    # lives, so nothing else needs to load .env separately here.
    from diqduq_main import _configure_lm as configure_lm

    from diqduq import (
        print_analysis,
        analyze_passage,
        tokengraph_to_mermaid,
        combined_tokengraph,
        tokengraph_to_html,
        tokengraph_to_text,
        tokengraph_to_depth_html,
        serialize_analyses,
    )

    return (
        analyze_passage,
        combined_tokengraph,
        configure_lm,
        serialize_analyses,
        tokengraph_to_depth_html,
        tokengraph_to_html,
        tokengraph_to_mermaid,
        tokengraph_to_text,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Configuration of LM
    """)
    return


@app.cell
def _(configure_lm, dspy):
    # Avoid reconfiguring the LM (and re-reading .env) on every reactive
    # re-run of this cell -- configure_lm() (diqduq_main._configure_lm())
    # raises with a clear message if MODEL/API_KEY aren't set in .env; see
    # notes/USAGE.md's "Running an analysis from the command line".
    lm = dspy.settings.lm if dspy.settings.lm is not None else configure_lm()
    return (lm,)


if __name__ == "__main__":
    app.run()
