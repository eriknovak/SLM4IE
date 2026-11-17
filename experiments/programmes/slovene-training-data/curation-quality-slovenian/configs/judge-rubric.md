You are annotating documents drawn from a Slovenian pretraining corpus. Judge
each document on its own merits. You are not told which filter saw it or what
that filter decided, and you must not guess: your labels are the yardstick those
decisions are measured against.

Each document is the **first 2,000 characters** of a longer text, so it may stop
mid-sentence. Judge what is present; never penalise a document for ending
abruptly.

## Dimensions

Label every document on all seven dimensions.

1. **language** — `sl` if the document is mainly Slovenian, otherwise the ISO
   639-1 code of the language that dominates it (`en`, `hr`, `sr`, `de`, ...),
   or `mixed` when no single language holds a clear majority. Quoted foreign
   passages inside Slovenian prose still count as `sl`.
2. **text_type** — what the text *is*:
   - `prose` — connected sentences meant to be read: articles, transcripts,
     reports, fiction, forum posts, instructions.
   - `boilerplate` — navigation, cookie notices, headers and footers, contact
     blocks, repeated site furniture with little content of its own.
   - `list` — tables, catalogues, search results, indexes, price lists: real
     information, but not sentences.
   - `code` — source code, markup, configuration, structured data dumps.
   - `garbage` — encoding damage, OCR noise, random character sequences,
     truncated fragments that carry no readable meaning.
3. **adult_or_spam** — `true` for pornographic content, gambling or pharmacy
   spam, SEO keyword stuffing, scam and "get rich" pitches. Ordinary
   advertising inside otherwise real content is `false`. Clinical or academic
   discussion of sexuality is `false`.
4. **coherence** — 1 to 5, how well the text holds together as a piece of
   writing. Decide by asking, in order:
   - **Is any of it readable as language?** No → **1**.
   - **Do whole sentences outnumber fragments?** No → **2**.
   - **Could this be published as-is, after a proofread?** No → **3**. Use 3
     when the text is understandable but carries damage: sentences broken
     mid-way, paragraphs welded together, stray markup or navigation text
     between them, OCR slips.
   - **Count the flaws in what you can see.** A flaw is a typo, a wrong ending,
     an awkward construction, a punctuation error, a formatting artefact.
     - **5** — you can find **no** flaw. Clean throughout.
     - **4** — you can find **one or more**, but they are local: the text
       around them is intact and the meaning is never in doubt.
   The 4/5 line is the one that matters: do not award 5 to text you would edit.
   A single typo makes it a 4. If you hesitate between 4 and 5, it is a 4.
   Judge the writing, not the subject matter, and not whether you agree with it.
   A well-written list is still a list: score its entries as written.
5. **machine_translated** — `true` when the Slovene reads as unedited machine
   output: literal word order from another language, untranslated terms left in
   place, wrong case endings on otherwise fluent phrases. Use `false` when
   unsure; this flag is only useful when it is confident.
6. **pii** — `true` when the text exposes a private individual's personal data:
   full name together with address, phone number, email, national ID, bank
   account, health record, or similar. Public figures acting in their public
   role, company contact details and parliamentary speakers on the record are
   `false`.
7. **domain** — the single best fit from: `medical`, `scientific`, `legal`,
   `news`, `parliamentary`, `academic`, `wiki`, `forum`, `blog`, `student`,
   `web`, `finance`, `other`. Use `web` for general-purpose web text that fits
   no other label, and `other` only when the text belongs to none of them.

## Judging

- A document can be perfectly good and still be `boilerplate` or `list`; those
  labels describe its form, not its worth.
- Do not reward length. A short, clean paragraph outranks a long, broken one.
- Slovene dialect, older orthography and regional usage are correct Slovene, not
  errors.
- Missing diacritics (č, š, ž written as c, s, z) lower coherence by at most one
  point; they are common in older web text and do not make it unreadable.

## Output

Return **one JSON array and nothing else** — no explanation, no code fence. One
object per input document, in the same order, with the same `id`:

```json
[
  {
    "id": "<the document's id, copied exactly>",
    "language": "sl",
    "text_type": "prose",
    "adult_or_spam": false,
    "coherence": 4,
    "machine_translated": false,
    "pii": false,
    "domain": "news",
    "note": ""
  }
]
```

`note` is at most one short sentence, and only when something about the document
would otherwise be invisible in the labels (for example: "truncated mid-table",
"legal text inside a news report"). Leave it empty otherwise.

## Documents

{{DOCUMENTS}}
