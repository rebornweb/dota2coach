# Item popularity and experimental ML

The app calls `GET /item-popularity?hero=Anti-Mage`. The backend resolves the
selected hero to its OpenDota ID, requests `/heroes/{id}/itemPopularity`, and
resolves item IDs through `/constants/items`. The UI checks once a minute;
the backend caches popularity for five minutes and item names for one day.
Failed refreshes use stale cached data with a visible warning and a one-minute
retry delay. Timestamps show when this app fetched the data, not when OpenDota
updated its underlying matches. Empty stages mean no reported purchases.

Item/build questions also receive the current hero's top ten purchases per
stage in the Ollama context. The existing system prompt remains in use.
Popularity supplies names and purchase counts, not item effects. There is no
game-mode filter, so these are not Turbo-specific statistics.

From `dota-ai-backend`, using the same Python environment as the backend:

```powershell
py ml/train_item_popularity.py
```

This collects all heroes with a two-second pause between hero requests, saves
`ml/item_popularity_training.json`, trains a random forest, and writes
`ml/item_popularity.joblib`. API errors stop collection rather than silently
training on missing heroes. Completed requests remain cached for a retry.
You can limit collection (at least five heroes with data are needed):

```powershell
py ml/train_item_popularity.py --hero-ids 1 2 3 4 5 6
py ml/train_item_popularity.py --dataset ml/item_popularity_training.json
py ml/train_item_popularity.py --predict 1 --stage mid_game_items
```

The model takes hero ID, stage, and item ID and estimates percentage of that
stage's purchases. Candidate items are those observed in any stage for that
hero; absent candidates in a stage get zero labels. Shares are purchase
shares, not the percentage of players buying an item. Predictions are
independent and may not sum to 100%. The model cannot discover unseen items.

Evaluation holds out entire heroes and prints mean absolute error in percentage
points, alongside a mean-per-item/stage baseline. Compare these before using
the estimator; it may perform worse than the baseline. It is refit on all
collected heroes after evaluation. This tests transfer across heroes within
one collection, not performance on future patches. A single aggregate snapshot
does not establish trends, optimal builds, win rates, or matchup advice.

The model is an optional standalone experiment. The app displays actual
OpenDota counts and does not replace them with model estimates. No new model
is required to start the app. To retrain later, rerun collection after the
cache expires. Respect your OpenDota plan's request quota and rate limits.

References: [OpenDota API](https://docs.opendota.com/#tag/heroes/operation/get_heroes_by_hero_id_select_item_popularity),
[OpenDota schemas](https://opendota.apidocumentation.com/).
