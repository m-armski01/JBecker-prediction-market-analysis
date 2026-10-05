# Holdout unlock log (append-only)

`unlock.py` appends one entry per holdout unlock and refuses to run a holdout that already has an entry. Entries are never edited or removed.

Entry format: `## Holdout <A|B>`, followed by the UTC timestamp, the HEAD commit hash, the SHA-256 of the panel manifest, and the line "unlocked once".

<!-- entries below this line -->
