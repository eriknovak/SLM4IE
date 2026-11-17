You are comparing pairs of documents drawn from a Slovenian pretraining corpus.
For each pair, say which of the two texts is the better material for training a
Slovenian language model. Both documents of a pair come from the same source and
are shown to you in an arbitrary order; nothing distinguishes them but the text
itself.

Each document is the **first 2,000 characters** of a longer text, so it may stop
mid-sentence. Judge what is present; never penalise a document for ending
abruptly.

## What makes a text better

Weigh these in order — an earlier point outranks every later one:

1. **Slovene.** Text mainly in Slovenian beats text in another language or a
   mixture with no clear majority. Quoted foreign passages inside Slovenian
   prose still count as Slovene.
2. **Readable.** Text a person can read beats encoding damage, OCR noise,
   random character sequences and fragments carrying no meaning.
3. **Written, not furniture.** Connected sentences meant to be read — articles,
   transcripts, reports, fiction, forum posts, instructions — beat navigation
   menus, cookie notices, contact blocks, repeated site furniture, and endless
   catalogues or price lists. A table of real information still ranks below
   prose, but above boilerplate.
4. **Clean.** Fewer typos, broken sentences, welded paragraphs, stray markup
   and wrong endings is better.
5. **Not spam.** Pornography, gambling and pharmacy spam, keyword stuffing and
   scam pitches are worse than any ordinary text. Ordinary advertising inside
   real content is not spam.

Judge the writing, not the subject matter, and not whether you agree with it.
Slovene dialect, older orthography and regional usage are correct Slovene, not
errors; missing diacritics (č, š, ž written as c, s, z) are a minor flaw.
Do not reward length: a short clean paragraph beats a long broken one.

Answer `tie` only when the two texts are genuinely of the same standard — when
both are good, or both are equally unusable. If one is even slightly better on
the first point where they differ, name it.

## Output

Return **one JSON array and nothing else** — no explanation, no code fence. One
object per input pair, in the same order, with the same `id`:

```json
[
  {
    "id": "<the pair's id, copied exactly>",
    "better": "a",
    "note": ""
  }
]
```

`better` is `a`, `b` or `tie`. `note` is at most one short sentence, and only
when the reason would otherwise be invisible (for example: "b is Croatian",
"both are navigation menus"). Leave it empty otherwise.

## Pairs

{{DOCUMENTS}}
