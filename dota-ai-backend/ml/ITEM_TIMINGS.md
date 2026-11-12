# Scenario integration

`GET /scenarios?hero=Dazzle` resolves the localized hero name to its numeric ID
and returns item timings, lane roles, and global team scenarios. Each dataset
includes a source URL, fetch timestamp, stale flag, and independent error state.
The cache expires after one hour and falls back to stale validated data during
outages. No game-mode filter is available; these are not Turbo-specific results.

The UI fetches on hero changes and checks every five minutes. Item/build,
lane/role, and supported team-scenario questions receive relevant data in the
current Ollama context, limited to the 20 largest samples per dataset. The
original system prompt and existing models remain in use. Casual questions
do not request scenario data. Missing rows in the context do not imply zero games.

Item `time` is a purchase cutoff in seconds, not an exact purchase minute.
Lane-role `time` is maximum match duration. Lane roles mean safe/mid/off/jungle,
not positions 1 through 5. The UI displays games, wins, observed win percentage,
and a small-sample flag below 100 games. Thresholds overlap; never add their
game counts. Outcomes are associations and do not prove an item causes wins.

## Optional ML

From `dota-ai-backend`, with the backend Python environment:

```powershell
py ml/train_item_timings.py
py ml/train_item_timings.py --dataset ml/item_timings_training.json
py ml/train_item_timings.py --predict 50 --item spirit_vessel --time 900
```

The first command retrieves all hero/item timing rows in one request, saves a
raw snapshot, and creates `ml/item_timings.joblib`. Training defaults to at least
100 games per row (`--min-games` changes this). Unknown/filtered heroes or items
are rejected at prediction time. Hero and item are categorical inputs; time is
numeric seconds. Target is wins/games, weighted by games. Counts are not input
features at inference. Duplicate hero/item/time rows are rejected.

Evaluation holds out whole hero/item pairs so overlapping timing thresholds
for a pair cannot appear in both sets. It reports game-weighted MAE in percentage
points and compares against the training weighted-mean baseline, then refits on
all data. Overlap between different pairs and repeated matches still exists:
this is not an independent match-level test or future-patch validation. Larger
weights do not turn correlated aggregate counts into independent observations.

The saved model is optional and standalone. The app presents actual scenario
counts and observed rates; it does not silently substitute model estimates.
Training neither replaces the existing popularity model nor runs during chats.
Compare the reported baseline before using predictions, and retain the snapshot
for reproducibility. These aggregates cannot control for skill, draft, gold
advantage, patch, or match state.
