# Team Collaboration Rulebook

## Branch Structure
- `main` — production-ready code only. Never push directly here.
- `dev` — shared integration branch. Merge your feature branch here first.
- `feature/your-name/feature-name` — your personal working branch.

## Your Branch Naming
- karnica/xbrl-loader
- teammateA/html-parser
- teammateB/book-rag

## Daily Workflow

### Starting work
git checkout dev
git pull origin dev
git checkout -b karnica/your-feature-name


### While working
# Save progress frequently
git add .
git commit -m "type: short description"


### Finishing a feature
git checkout dev
git pull origin dev
git checkout karnica/your-feature-name
git merge dev  # bring in any new teammate changes
# fix conflicts if any
git push origin karnica/your-feature-name
# then open a PR on GitHub: your branch → dev


## Commit Message Format
Always use this format:
feat: add revenue tag normalization
fix: handle null values in num.txt parser
docs: update README with setup instructions
test: add acceptance query for Apple revenue
refactor: simplify bulk insert logic


## PR Rules
1. Never open a PR directly to `main` — always go to `dev` first
2. Every PR needs at least 1 approval from a teammate
3. Your PR → A or B reviews
4. A's PR → you or B reviews  
5. B's PR → you or A reviews
6. B (main owner) merges dev → main at end of each week
7. Delete your branch after it's merged

## What Goes in a PR Description
- What you built
- How to test it
- Any known issues
- Screenshots or query output as proof it works

## What Never Goes in Git
- Raw SEC data files (num.txt, sub.txt) — too large
- .venv/ folder
- .env files with secrets
- Any file over 50MB

## Weekly Merge Schedule
- Every Friday: B reviews dev, merges to main if stable
- Tag the release: git tag -a v0.1 -m "Week 1 complete"

## Conflict Resolution
If two people edited the same file:
1. Don't panic
2. Open the file — git marks conflicts with <<<<<<< and >>>>>>>
3. Pick the right version, delete the markers
4. Commit the resolved file