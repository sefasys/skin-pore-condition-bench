# Contributing

Thank you for improving this research pipeline.

1. Open an issue describing the change and its scientific motivation.
2. Create a focused branch.
3. Add or update tests for behavior changes.
4. Run `pytest` and `ruff check .`.
5. Do not commit dataset images, face-containing figures, prediction tables
   with restricted filenames, checkpoints, caches, credentials, or access
   agreements.
6. Clearly label exploratory results and keep held-out test data locked.

Changes to preprocessing, splits, augmentation, architecture, loss, checkpoint
selection, or metrics define a new experimental protocol. Record them in a new
configuration file rather than silently changing the reported final protocol.

By contributing code, you agree that it may be distributed under the MIT
License. You are responsible for ensuring you have the right to share all
submitted material.

