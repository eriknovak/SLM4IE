# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

## Kind labels

Beside the triage roles, every issue carries one kind label saying which
workflow owns it:

| Label  | Meaning                                                              |
| ------ | -------------------------------------------------------------------- |
| `dev`  | Development task — delivery work (devflow)                           |
| `lab`  | Experiment task — research work (labflow)                            |
| `idea` | A change to a reference entry that could become an experiment; always paired with `lab` (labflow:reference) |

## Experiment labels

A `lab` issue filed by `labflow:start` also carries its category and its
line, mirroring the record's frontmatter; a human-filed one gets them at
triage:

| Group    | Labels                              | Meaning                                       |
| -------- | ----------------------------------- | --------------------------------------------- |
| Category | `data` / `methods` / `validation`   | What the hypothesis is about                  |
| Line     | `main-line` / `secondary`           | Merges into main, or record merges and branch freezes |

The same labels, plus `experiment`, mark the conclusion PR.

Edit the right-hand column to match whatever vocabulary you actually use.
