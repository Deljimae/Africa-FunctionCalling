# Development tasks

This folder contains the 12 English tasks used to develop and test the inference
and scoring workflow. They are not part of the main evaluation sample.

- `tasks.jsonl` contains the requests, candidate tools and reference answers.
  The inference runner sends only requests and tool definitions to the model;
  references are used separately for scoring.
- `manifest.json` records the source revision, selected IDs, selection procedure,
  file hashes, development exclusions and the remaining 188 candidate IDs.

The manifest preserves the original development-pack record. Its `files` section
also lists hashes of three historical blank language-review CSVs. Those forms
are not included in this public snapshot and are not required by the runner or
tests. They are not completed reviews or accepted translations.

The remaining 188 IDs are candidates, not a finalized test set. The manifest also
records duplicate source requests; final inclusion and reporting decisions will
be documented before main evaluation.
