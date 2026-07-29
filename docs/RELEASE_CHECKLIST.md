# Release Checklist

Before creating the public GitHub repository:

- [ ] Replace `YOUR_USERNAME` in `CITATION.cff` and the Colab notebook.
- [ ] Replace the placeholder contributor name in `CITATION.cff` and `LICENSE`
  with the preferred author name(s).
- [ ] Confirm that the Zenodo access agreement permits the intended use.
- [ ] Do not add the dataset to Git, GitHub Releases, Kaggle, or another public
  host without explicit written redistribution permission.
- [ ] Run `pytest` and `ruff check .`.
- [ ] Run `pore-pipeline --config configs/final_protocol.json --dry-run`.
- [ ] Inspect `git status` and check for images, archives, model weights,
  credentials, private filenames, and local absolute paths.
- [ ] Keep the current outputs labeled as validation-only.
- [ ] Add the final GitHub URL to `CITATION.cff`.
- [ ] Create a tagged release after the first push.

Suggested commands:

```bash
git init
git add .
git status
git commit -m "Initial reproducible pore-condition study"
git branch -M main
git remote add origin <YOUR-GITHUB-REPOSITORY-URL>
git push -u origin main
```

These commands are intentionally not run automatically; the repository owner
should review the release contents and choose the remote.
