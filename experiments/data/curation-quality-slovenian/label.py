"""Hand-label the calibration set for the curation-quality experiment.

Run it with:

    uv run marimo edit experiments/data/curation-quality-slovenian/label.py

The judge's labels only become evidence once their agreement with a person is
known, so this notebook collects that person's labels on the same rubric. The
document sits on the left with the rules beside it, the labels are typed as a
short command, and Enter saves them and moves on. Alt-N and Alt-P move without
labelling. Closing the notebook loses nothing — it resumes at the first
unlabelled document.

The judge's verdict stays hidden until the label is saved, and then shows only
where the two differ. Revealing it earlier would anchor the labeller, and the
agreement the pair produced would measure nothing.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def imports():
    """Load the libraries and the files this notebook reads and writes.

    Returns:
        The marimo module, the json and html modules, and the three paths.
    """
    import html
    import json
    from pathlib import Path

    import marimo as mo

    interim = Path("data/experiments/data/curation-quality-slovenian/interim")
    calibration_path = interim / "calibration.jsonl"
    labels_path = interim / "human-labels.jsonl"
    verdicts_path = interim / "verdicts-sonnet.jsonl"
    return calibration_path, html, json, labels_path, mo, verdicts_path


@app.cell
def vocabulary():
    """Define what each typed token may mean.

    Returns:
        The label vocabularies and the shorthand that selects each value.
    """
    text_types = {"p": "prose", "b": "boilerplate", "l": "list", "c": "code", "g": "garbage"}
    domains = {
        "med": "medical",
        "sci": "scientific",
        "leg": "legal",
        "new": "news",
        "par": "parliamentary",
        "aca": "academic",
        "wik": "wiki",
        "for": "forum",
        "blo": "blog",
        "stu": "student",
        "web": "web",
        "fin": "finance",
        "oth": "other",
    }
    languages = ("sl", "en", "hr", "sr", "de", "it", "hu", "mixed")
    flags = {"!s": "adult_or_spam", "!m": "machine_translated", "!p": "pii"}
    return domains, flags, languages, text_types


@app.cell
def loading(calibration_path, json, labels_path, verdicts_path):
    """Read the documents, the labels given so far, and the judge's verdicts.

    Args:
        calibration_path: JSONL of documents to label.
        json: The json module.
        labels_path: JSONL of labels given so far.
        verdicts_path: JSONL of the judge's verdicts.

    Returns:
        The documents and a reader for JSONL files.
    """

    def read_jsonl(path):
        """Read a JSONL file, splitting on newline only.

        Args:
            path: The file to read.

        Returns:
            One dict per line; empty when the file does not exist.
        """
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]

    documents = read_jsonl(calibration_path)
    verdicts = {row["id"]: row for row in read_jsonl(verdicts_path)}
    labelled = {row["id"] for row in read_jsonl(labels_path)}
    start_at = next((i for i, doc in enumerate(documents) if doc["id"] not in labelled), len(documents))
    return documents, read_jsonl, start_at, verdicts


@app.cell
def position(documents, mo, start_at):
    """Track which document is on screen and move between them.

    Args:
        documents: The calibration set.
        mo: The marimo module.
        start_at: Index of the first unlabelled document.

    Returns:
        The index state, its setter, and a bounded step function.
    """
    # allow_self_loops: Enter is handled by an element `screen` defines itself,
    # and marimo does not re-run the setter's own cell without this.
    get_index, set_index = mo.state(start_at, allow_self_loops=True)

    def step(delta):
        """Move by `delta` documents, stopping at either end.

        Args:
            delta: How far to move; negative goes back.
        """
        set_index(lambda i: min(len(documents), max(0, i + delta)))

    return get_index, set_index, step


@app.cell
def parsing(domains, flags, languages, text_types):
    """Turn a typed command into a set of labels.

    Args:
        domains: Shorthand to domain name.
        flags: Shorthand to flag name.
        languages: Accepted language codes.
        text_types: Shorthand to text-type name.

    Returns:
        The parser.
    """

    def parse_command(command):
        """Read one typed command into labels.

        Args:
            command: What the labeller typed, e.g. `4 news` or `2 g !s`.

        Returns:
            Tuple `(labels, error)`; `labels` is None when `error` is set.

        """
        labels = {
            "language": "sl",
            "text_type": "prose",
            "coherence": None,
            "domain": None,
            "adult_or_spam": False,
            "machine_translated": False,
            "pii": False,
        }
        for token in command.lower().split():
            if token in flags:
                labels[flags[token]] = True
            elif token.isdigit() and token in "12345" and len(token) == 1:
                labels["coherence"] = int(token)
            elif token in text_types:
                labels["text_type"] = text_types[token]
            elif token in text_types.values():
                labels["text_type"] = token
            elif token in languages:
                labels["language"] = token
            elif token[:3] in domains:
                labels["domain"] = domains[token[:3]]
            else:
                return None, f"`{token}` means nothing here"
        if labels["coherence"] is None:
            return None, "give a coherence score, 1 to 5"
        if labels["domain"] is None:
            return None, "give a domain, e.g. `news` or `med`"
        return labels, ""

    return (parse_command,)


@app.cell
def saving(json, labels_path, mo, parse_command, step):
    """Save a typed label and move to the next document.

    Args:
        json: The json module.
        labels_path: JSONL file the labels are appended to.
        mo: The marimo module.
        parse_command: The command parser.
        step: Moves the index by a bounded number of documents.

    Returns:
        The commit handler and the message state it writes to.
    """
    get_message, set_message = mo.state("", allow_self_loops=True)

    def commit(document, command):
        """Parse, append and advance, or report why the command was refused.

        Args:
            document: The document on screen.
            command: What the labeller typed.
        """
        text = (command or "").strip()
        if not text or document is None:
            return
        if text in {"skip", "s", "next", "n"}:
            set_message("moved on, nothing saved")
            step(1)
            return
        if text in {"back", "b", "prev", "p"}:
            set_message("")
            step(-1)
            return
        labels, error = parse_command(text)
        if error:
            set_message(f"**{error}** — nothing saved")
            return
        row = {"id": document["id"], **labels}
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        set_message(f"saved `{row['coherence']} {row['text_type']} {row['domain']}`")
        step(1)

    return commit, get_message, set_message


@app.cell
def navigation(mo, set_message, step):
    """Build the buttons that move between documents without labelling.

    Args:
        mo: The marimo module.
        set_message: Setter for the line beside the counter.
        step: Moves the index by a bounded number of documents.

    Returns:
        The previous and next buttons.
    """

    def move(delta):
        """Move without saving anything.

        Args:
            delta: How far to move; negative goes back.
        """
        set_message("")
        step(delta)

    # Alt-P / Alt-N: free in both Firefox and Chrome, and they still fire while
    # the cursor sits in the command box.
    previous_button = mo.ui.button(
        label="← previous",
        tooltip="Alt-P",
        on_click=lambda _: move(-1),
        keyboard_shortcut="Alt-P",
    )
    next_button = mo.ui.button(
        label="next →",
        tooltip="Alt-N",
        on_click=lambda _: move(1),
        keyboard_shortcut="Alt-N",
    )
    return move, next_button, previous_button


@app.cell
def instructions(mo):
    """Show how to type a label, and the rubric behind each dimension.

    Args:
        mo: The marimo module.

    Returns:
        The always-visible cheat sheet and the two collapsible explanations.
    """
    cheat_sheet = mo.md(
        """
        **Type the labels in any order, press Enter.** Enter saves the label and
        moves to the next document. Two labels are required — a coherence score
        and a domain — the rest have a default. To move without labelling, use
        the buttons under the box, or **Alt-N** forward and **Alt-P** back.

        | Type | Gives | Default |
        | --- | --- | --- |
        | `1` `2` `3` `4` `5` | how well the text holds together, 1 worst to 5 clean | **required** |
        | a domain code | what the text is about — full words like `news` work too | **required** |
        | `p` `b` `l` `c` `g` | prose · boilerplate · list · code · garbage | `prose` |
        | `en` `hr` `sr` `de` `it` `hu` `mixed` | the language, when it is not Slovene | `sl` |
        | `!s` `!m` `!p` | adult-or-spam · machine-translated · contains personal data | all off |
        | `skip` or `n` | next document, nothing saved — same as **Alt-N** | |
        | `back` or `p` | back one document — same as **Alt-P** | |

        Domains: `med` `sci` `leg` `new` `par` `aca` `wik` `for` `blo` `stu`
        `web` `fin` `oth` — what each one covers is in the panel below.

        `4 news` — a readable Slovene news article.
        `2 g web !s` — near-unreadable Slovene web garbage that is also spam.
        `5 med en` — a flawless English medical text.
        """
    )
    dimensions = mo.md(
        """
        **coherence, `1`–`5`** — judge the writing, not the subject. Ask in order:

        - Is *any* of it readable as language? No → **1**.
        - Do whole sentences outnumber fragments? No → **2**.
        - Would it need more than a proofread? Yes → **3**. This is understandable
          text that carries damage: sentences cut mid-way, paragraphs welded
          together, menus or markup left between them, OCR slips.
        - Otherwise count the flaws — a typo, a wrong ending, an awkward
          construction, a stray bit of formatting. **5** only when you find
          *none*. **4** when you find some but they are local and the meaning is
          never in doubt.

        A single typo makes it a **4**. Hesitating between 4 and 5 means **4**.

        **text type** — what the text *is*, never how good it is. A clean
        price list is still a `list`.

        - `p` **prose** — connected sentences meant to be read: articles,
          transcripts, reports, fiction, forum posts, instructions.
        - `b` **boilerplate** — site furniture: menus, cookie notices, headers
          and footers, contact blocks, anything repeated across pages.
        - `l` **list** — tables, catalogues, search results, indexes, price
          lists. Real information, but not sentences.
        - `c` **code** — source code, markup, configuration, data dumps.
        - `g` **garbage** — encoding damage, OCR noise, random characters,
          fragments carrying no readable meaning.

        **language** — the language that dominates the text. Quoted foreign
        passages inside Slovene prose are still `sl`; `mixed` is for text where
        no single language holds a clear majority.

        **`!s` adult or spam** — pornography, gambling or pharmacy spam, SEO
        keyword stuffing, scams and get-rich pitches. Ordinary advertising
        inside otherwise real content is not spam, and clinical or academic
        writing about sexuality is not adult.

        **`!m` machine-translated** — the Slovene reads as unedited machine
        output: word order carried over from another language, terms left
        untranslated, wrong case endings on otherwise fluent phrases. Only flag
        it when you are confident.

        **`!p` personal data** — a private individual is identifiable: full name
        together with an address, phone number, email, ID or bank number, or a
        health record. Public figures acting publicly, company contact details
        and named parliamentary speakers are not.

        Dialect, older orthography and regional usage are correct Slovene.
        Missing diacritics (c, s, z for č, š, ž) cost at most one coherence point.
        """
    )
    domain_guide = mo.md(
        """
        Pick the single best fit. Type three letters or the whole word.

        | Type | Domain | What belongs here |
        | --- | --- | --- |
        | `med` | medical | clinical notes, patient leaflets, drug information, health advice, medical research |
        | `sci` | scientific | research and technical writing outside medicine: sciences, engineering, computing |
        | `leg` | legal | laws, regulations, contracts, court decisions, official gazettes, terms and conditions |
        | `new` | news | journalism: reports, interviews, press releases, sports and weather coverage |
        | `par` | parliamentary | debate in parliament or council, transcripts of sittings, records of speech |
        | `aca` | academic | university-level scholarly work: theses, dissertations, lecture material, textbooks |
        | `wik` | wiki | encyclopedia articles and other collaboratively edited reference text |
        | `for` | forum | discussion threads, comment sections, question-and-answer exchanges |
        | `blo` | blog | personal or opinion writing published under an author's own voice |
        | `stu` | student | writing by school pupils and language learners, typically with learner errors |
        | `web` | web | general web pages that fit none of the above: company pages, product copy, portals |
        | `fin` | finance | banking, markets, company reports, accounting, tax |
        | `oth` | other | none of the above genuinely fits |

        Two pairs are easy to confuse:

        - **academic vs scientific** — `aca` is the *setting* (a thesis, a
          seminar paper, coursework), `sci` is the *content* (a research result,
          a technical description). A physics dissertation is `aca`; a physics
          paper in a journal is `sci`.
        - **academic vs student** — `stu` is school-level or learner writing.
          If it reads as an exercise rather than scholarship, it is `stu`.
        """
    )
    # Rendered by `screen` as the right-hand column, so the document and the
    # rules it is judged against are on the same screenful.
    guide = mo.vstack(
        [
            cheat_sheet,
            mo.accordion({"What each label means": dimensions, "Which domain to pick": domain_guide}),
        ],
        gap=0.5,
    )
    return cheat_sheet, dimensions, domain_guide, guide


@app.cell
def screen(
    commit,
    documents,
    get_index,
    get_message,
    guide,
    html,
    mo,
    next_button,
    previous_button,
    verdicts,
):
    """Show the document and command box beside the labelling instructions.

    Args:
        commit: The save handler.
        documents: The calibration set.
        get_index: Index state.
        get_message: Message state.
        guide: The instructions column.
        html: The html module, for escaping document text.
        mo: The marimo module.
        next_button: Moves to the next document without saving.
        previous_button: Moves back one document.
        verdicts: The judge's verdicts, keyed by id.

    Returns:
        The rendered screen.
    """
    index = get_index()
    total = len(documents)
    if index >= total:
        screen_view = mo.vstack(
            [mo.md(f"## Done — all {total} documents seen."), previous_button],
            gap=0.8,
        )
    else:
        document = documents[index]
        # Serif, generous line height: diacritics and long compounds are what is
        # being judged, so the text has to be comfortable to read closely.
        body = mo.Html(
            '<div style=\'font-family:Georgia,"Noto Serif",serif;font-size:1.06rem;line-height:1.75;'
            "white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;"
            "width:100%;max-width:100%;box-sizing:border-box;"
            "height:62vh;overflow-y:auto;overflow-x:hidden;padding:1.1rem 1.3rem;"
            "border:1px solid #dcdcdc;border-radius:8px;background:#fffdf9;color:#1a1a1a'>"
            f"{html.escape(document['text'])}</div>"
        )
        bar = mo.ui.text(
            placeholder="e.g.  4 news    ·    2 g web !s    ·    then press Enter",
            debounce=True,
            full_width=True,
            on_change=lambda value: commit(document, value),
        )
        controls = mo.hstack(
            [
                mo.hstack([previous_button, next_button], gap=0.5, justify="start"),
                mo.md("<small>Enter saves and moves on · Alt-P back · Alt-N forward</small>"),
            ],
            justify="space-between",
            align="center",
        )
        filled = round(100 * index / total)
        meter = mo.Html(
            f"<div style='height:6px;background:#ececec;border-radius:3px;overflow:hidden'>"
            f"<div style='height:6px;width:{filled}%;background:#4a7fb5'></div></div>"
        )
        last = get_message()
        reading = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.md(f"**{index} / {total}**  ·  `{document['id']}`  ·  {document['chars']:,} chars"),
                        mo.md(last),
                    ],
                    justify="space-between",
                ),
                meter,
                body,
                bar,
                controls,
            ],
            gap=0.6,
        )
        screen_view = mo.hstack([reading, guide], widths=[3, 2], gap=1.6, align="start")
    screen_view
    return (screen_view,)


@app.cell
def previous(documents, get_index, get_message, mo, verdicts):
    """Show where the judge disagreed with the label just saved.

    Args:
        documents: The calibration set.
        get_index: Index state.
        get_message: Message state.
        mo: The marimo module.
        verdicts: The judge's verdicts, keyed by id.

    Returns:
        The comparison line.
    """
    position_now = get_index()
    comparison = mo.md("")
    if get_message().startswith("saved") and 0 < position_now <= len(documents):
        judged = verdicts.get(documents[position_now - 1]["id"])
        if judged:
            summary = " · ".join(
                f"{field} **{judged[field]}**" for field in ("language", "text_type", "coherence", "domain")
            )
            comparison = mo.md(f"<small>judge on the previous document: {summary}</small>")
    comparison
    return (comparison,)


if __name__ == "__main__":
    app.run()
