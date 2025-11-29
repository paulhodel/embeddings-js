# Repository Guidelines

## Project Structure & Module Organization
- `src/`: ES module code for CSS; key scripts include `train.js` (trainer entry), `prepare_vocabulary.js` (vocab initialization), `download_parquet.js` (data fetch), `analyze_polysemy.js` / `analyze_stability.js` (analysis), `debug.js` (troubleshooting), and `index.js` (demo).
- `data/`: Local artifacts; Parquet corpora (`data/parquet/`), training snapshots (`data/snapshots/`), and generated vocab/state JSON. Kept out of Git via `.gitignore`—do not commit large data.
- `scripts/`: Auxiliary helpers such as `read_parquet.py` for inspection.
- `docs/`, `README.md`, `QUICKSTART.md`: Concept notes and step-by-step guides; consult before changing training flow.

## Build, Test, and Development Commands
- Install deps: `npm install`
- Demo run: `npm start` (tokenize small corpus, train, report spectra/similarity)
- Data fetch: `npm run download` (downloads HuggingFace Parquet into `data/parquet/`)
- Vocab prep: `npm run prepare:frequency-scaled` (recommended), or `prepare:gaussian` / `prepare:uniform`
- Training: `npm run train` (uses increased heap: `--max-old-space-size=12288`)
- Analysis: `npm run analyze -- <snapshot> <target> [comparison]` or `npm run analyze:all`
- Stability/debug: `npm run stability`, `npm run debug`
- Basic test harness: `npm test` (executes `src/test.js`)

## Coding Style & Naming Conventions
- JavaScript ES modules; prefer 4-space indentation and semicolons for consistency with existing files.
- PascalCase for classes (e.g., `Tokenizer`, `CSSTrainer`); camelCase for functions/variables; kebab/snake-case for filenames that mirror scripts (`prepare_vocabulary.js`).
- Keep functions small and pure where possible; document non-obvious math or data transformations with brief inline comments.

## Testing Guidelines
- Add runnable cases to `src/test.js` or nearby modules; favor fast checks that validate spectra shapes, sparsity, and tokenization edge cases.
- When touching training or analysis logic, run `npm test` plus the relevant analysis command against a small snapshot in `data/snapshots/` to confirm outputs still render.
- Include example commands in PRs for any new test or analysis path.

## Commit & Pull Request Guidelines
- Git history is terse (e.g., `review`); follow with short, present-tense subjects. If helpful, prefix scope (`train: adjust sparsity penalty`).
- PRs should describe intent, key changes, and validation (commands run, sample output). Link issues when applicable and note data requirements (e.g., Parquet location, snapshot size). Include screenshots/log excerpts for analysis output changes.

## Data & Operational Notes
- Large artifacts (`data/`, `models/`, `checkpoints/`, `venv/`) are ignored—keep them local. Avoid committing generated vocab/snapshot JSON.
- When adding new scripts, default paths under `data/` and accept overrides via CLI flags to keep experiments reproducible.
