# paper/

Drop the paper and any supporting material here before starting an implementation run.

## What to put here

| File | Description |
|------|-------------|
| `paper.pdf` | The full paper PDF |
| `supplement.pdf` | Supplementary material, if provided |
| `architecture.md` | (Optional) Hand-written architecture notes — useful if the paper PDF is hard to parse |
| `pseudocode.md` | (Optional) Key algorithms transcribed as pseudocode for clarity |
| `figures/` | (Optional) Architecture diagrams, exported from the paper |

## Guidelines

- **One paper per run.** If you want to implement a different paper, create a new branch.
- The agent reads this directory in Phase 0 (Setup). Everything here should be the authoritative source for what to implement.
- If the paper references an external codebase but does not release code, you may note the reference in `architecture.md` — but do not provide the code itself. The agent implements from the paper description.
- If there are ambiguities in the paper (e.g. unclear hyperparameters, missing algorithm steps), document them in `architecture.md` with your best interpretation. The agent will encounter these ambiguities too.
