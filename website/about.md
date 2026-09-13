---
title: About
template: about.html
subtitle: Motivation, focus and facts of the SLM4IE project.
description: Why SLM4IE exists, what the research focuses on, and the project's key facts.
hide:
  - navigation
facts:
  - label: Title
    value: Small language models for zero-shot information extraction in European languages
  - label: Project number
    value: <a href="https://cris.cobiss.net/ecris/si/sl/project/24346" target="_blank">Z2-70067</a>
  - label: Scheme
    value: ARIS postdoctoral research project
  - label: Funder
    value: <a href="https://www.aris-rs.si/" target="_blank">ARIS</a>, the Slovenian Research and Innovation Agency
  - label: Duration
    value: March 2026 – February 2028
  - label: Host
    value: <a href="https://ailab.ijs.si/" target="_blank">Department of Artificial Intelligence</a>, <a href="https://www.ijs.si/" target="_blank">Jožef Stefan Institute</a>
  - label: Partner
    value: <a href="https://eventregistry.org/" target="_blank">Event Registry</a>
  - label: Project leader
    value: <a href="https://cris.cobiss.net/ecris/si/sl/researcher/50358" target="_blank">dr. Erik Novak</a>
  - label: Contact
    value: <a href="mailto:erik.novak@ijs.si">erik.novak@ijs.si</a>
---

## Motivation

SLM4IE starts from three limits of today's large language models: running one
privately is expensive, the text that matters most was never in its training
data, and it does not answer the same way twice.

Sensitive material, e.g. medical, financial, legal, cannot be sent to a
proprietary cloud service, so the model has to run on hardware the organisation
owns, and that hardware costs more than most smaller institutions can justify.
The same material is absent from the web, so it is absent from the training
data too, and a model's sense of language drifts away from the terminology and
formatting of the field it is asked to read. Low-resource languages, Slovenian
among them, are thin in that training data for the same reason.

Generality has its own price. A broad model pays for its range on every task,
however small, in processing power, energy and cost. And because it generates
text, the same query can return a different answer on the next run. Information
extraction needs the opposite: the same answer every time, from a model that
fits the machine in the room.

## Project focus

The project builds small language models that do zero-shot information
extraction, meaning no labelled examples and no training run per task, and that
are small enough to serve on a commercial GPU. Both encoder-only and
decoder-only architectures are tried, and compression and optimisation
techniques are measured for the efficiency they buy against the accuracy they
cost. Alongside the models, the project curates benchmark datasets for domains
the existing benchmarks miss, so a small model can be compared with a much
larger one on the same ground. [The work packages](work-packages.md) set out
how the work is organised and what it delivers.

Results are published as they are made, not at the end: models and datasets on
[Hugging Face](https://huggingface.co/eriknovak), the data pipeline and the
training and evaluation code [on GitHub](https://github.com/eriknovak/SLM4IE),
each with the documentation needed to use it. Everything is released openly
wherever licensing allows.

## Project facts
