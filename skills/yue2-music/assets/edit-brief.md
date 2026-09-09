# Score-editing task

Goal: Reharmonize the supplied pop score as modern vocal jazz.

Inputs: original `score.abc`, `request.json`, and optional listening notes.

Constraints:
- Preserve Vocal pitch, onset and duration exactly unless the user permits changes.
- Keep native headers, voices, bar grouping, section comments and the ABC dialect.
- Keep lyrics and phrase lengths unless lyric adaptation is requested.
- Use piano, tenor saxophone, upright bass and brushed drums; omit guitars.
- Keep supplied tempo/meters unless changes are requested.
- Support sustained melody notes with compatible harmony; retain useful tension and resolve it.

Outputs in a new directory:
- `edited.abc`: complete score, including untouched passages.
- `request.json`: target style, lyrics, seed and cot; no unsupported API fields.
- `edit_manifest.json`: source hash, edited bars/voices, harmonic changes, permitted
  melody/rhythm/lyric changes, and preserved invariants.
- A short musical rationale tied to actual note spans and beat positions.

Connect the edited score explicitly: pass `--abc-file edited.abc` to the helper,
or put `"abc_path": "edited.abc"` in its request JSON. `abc_path` is a helper/CLI
convenience; the Python pipeline takes ABC text. A request with `abc: null` and
no external file will generate a new plan instead of using your edit.

Check the whole score. Do not report audio generation or listening if you only edited
notation. For a quoted theme, verify complete phrases and repetitions against the
source rather than repeating only its opening.
