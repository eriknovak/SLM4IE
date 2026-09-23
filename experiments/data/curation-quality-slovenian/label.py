"""Hand-label the calibration set for the curation-quality experiment.

Run it with:

    uv run marimo edit experiments/data/curation-quality-slovenian/label.py

The judge's labels only become evidence once their agreement with a person is
known, so this notebook collects that person's labels on the same rubric. One
document fills the screen; the labels are typed as a short command and Enter
moves on. Closing the notebook loses nothing — it resumes at the first
unlabelled document.

The judge's verdict stays hidden until the label is saved, and then shows only
where the two differ. Revealing it earlier would anchor the labeller, and the
agreement the pair produced would measure nothing.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


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
def position(mo, start_at):
    """Track which document is on screen.

    Args:
        mo: The marimo module.
        start_at: Index of the first unlabelled document.

    Returns:
        The index state and its setter.
    """
    get_index, set_index = mo.state(start_at)
    return get_index, set_index


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
def saving(json, labels_path, mo, parse_command, set_index):
    """Save a typed label and move to the next document.

    Args:
        json: The json module.
        labels_path: JSONL file the labels are appended to.
        mo: The marimo module.
        parse_command: The command parser.
        set_index: Setter for the index on screen.

    Returns:
        The commit handler and the message state it writes to.
    """
    get_message, set_message = mo.state("")

    def commit(document, command):
        """Parse, append and advance, or report why the command was refused.

        Args:
            document: The document on screen.
            command: What the labeller typed.
        """
        text = (command or "").strip()
        if not text or document is None:
            return
        if text in {"skip", "s"}:
            set_message("skipped")
            set_index(lambda i: i + 1)
            return
        if text in {"back", "b"}:
            set_message("")
            set_index(lambda i: max(0, i - 1))
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
        set_index(lambda i: i + 1)

    return commit, get_message, set_message


@app.cell
def instructions(mo):
    """Show how to type a label, and the rubric behind each dimension.

    Args:
        mo: The marimo module.

    Returns:
        The instructions block.
    """
    how_to = mo.md(
        """
        **Type the labels, press Enter.** Order does not matter, and everything
        except the score and the domain has a default.

        | Type | Meaning |
        | --- | --- |
        | `1`–`5` | coherence — **required** |
        | `med sci leg new par aca wik for blo stu web fin oth` | domain — **required**, first three letters are enough |
        | `p b l c g` | prose / boilerplate / list / code / garbage — defaults to prose |
        | `en hr sr de it hu mixed` | language, when it is not Slovene — defaults to `sl` |
        | `!s` `!m` `!p` | adult-or-spam, machine-translated, contains PII |
        | `skip` | move on without saving |
        | `back` | step back one document |

        So `4 news` is a readable Slovene news article. `2 g web !s` is
        near-unreadable web garbage that is also spam.
        """
    )
    rubric = mo.md(
        """
        **coherence** — ask in order: is any of it readable (no → 1); do whole
        sentences outnumber fragments (no → 2); could it be published after a
        proofread (no → 3); then **5** only if you can find *no* flaw, **4** if
        the flaws are local and the meaning is never in doubt. A single typo
        makes it a 4. Hesitating between 4 and 5 means 4.

        **text type** — what the text *is*, not how good it is. `prose` is
        connected sentences; `boilerplate` is navigation, cookie notices, site
        furniture; `list` is tables, catalogues, search results; `garbage` is
        encoding damage, OCR noise, unreadable fragments.

        **adult or spam** — pornography, gambling or pharmacy spam, SEO keyword
        stuffing, scams. Ordinary advertising inside real content is not.

        **machine-translated** — literal word order from another language,
        untranslated terms, wrong endings on fluent phrases. Use it only when
        confident.

        **PII** — a private individual's personal data. Public figures acting
        publicly, company contact details and named parliamentary speakers are
        not PII.

        Dialect, older orthography and regional usage are correct Slovene.
        Missing diacritics cost at most one coherence point.
        """
    )
    mo.accordion({"How to label": how_to, "What each label means": rubric})
    return how_to, rubric


@app.cell
def screen(commit, documents, get_index, get_message, html, mo, verdicts):
    """Show the document, the command box, and what the judge said about the last one.

    Args:
        commit: The save handler.
        documents: The calibration set.
        get_index: Index state.
        get_message: Message state.
        html: The html module, for escaping document text.
        mo: The marimo module.
        verdicts: The judge's verdicts, keyed by id.

    Returns:
        The rendered screen.
    """
    index = get_index()
    total = len(documents)
    if index >= total:
        screen_view = mo.md(f"## Done — all {total} documents labelled.")
    else:
        document = documents[index]
        # Serif, generous line height: diacritics and long compounds are what is
        # being judged, so the text has to be comfortable to read closely.
        body = mo.Html(
            '<div style=\'font-family:Georgia,"Noto Serif",serif;font-size:1.06rem;line-height:1.75;'
            "white-space:pre-wrap;height:24rem;overflow-y:auto;padding:1.1rem 1.3rem;"
            "border:1px solid #dcdcdc;border-radius:8px;background:#fffdf9;color:#1a1a1a'>"
            f"{html.escape(document['text'])}</div>"
        )
        bar = mo.ui.text(
            placeholder="e.g.  4 news    ·    2 g web !s    ·    skip",
            debounce=True,
            full_width=True,
            on_change=lambda value: commit(document, value),
        )
        filled = round(100 * index / total)
        meter = mo.Html(
            f"<div style='height:6px;background:#ececec;border-radius:3px;overflow:hidden'>"
            f"<div style='height:6px;width:{filled}%;background:#4a7fb5'></div></div>"
        )
        last = get_message()
        screen_view = mo.vstack(
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
            ],
            gap=0.6,
        )
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
