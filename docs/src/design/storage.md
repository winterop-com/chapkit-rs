# Storage: SQLite behind a trait

**Decision:** configs, artifacts and their links live behind the async `Store` trait in `chapkit-store`.
The default backend is SQLite through rusqlite (bundled), in memory unless `DATABASE_URL` is set.

**Why SQLite:** the service stores a handful of configs and artifacts and looks them up by id. That is a
small transactional workload, which SQLite is built for, and it matches what Python chapkit uses
(`data/chapkit.db`). One connection behind a mutex, run on tokio's blocking pool, is plenty: the fast paths
never touch the store.

**Why not DuckDB:** DuckDB is a columnar analytics engine. It is excellent for querying many prediction
frames, but it is a large C++ dependency, allows one writer, and is the wrong shape for key lookups. It may
fit later as an optional tool for analysing backtest outputs, not as the service store.

**Why not sqlx:** its async pool adds nothing at this size, and in-memory SQLite with a pool needs care
(each pooled connection is a separate database).

**turso:** turso reimplements SQLite in Rust and is already used in production elsewhere by the author.
It is available as a second backend behind the same trait: build with the `turso` feature and set
`DATABASE_URL=turso:///path/to.db` (or `turso::memory:`). It reads and writes SQLite's file format, so
moving between the two backends is cheap. The turso backend uses plain SQL only and does tree walks and
cascading deletes in code, so it does not depend on which SQLite features (recursive CTEs, foreign-key
actions) turso implements.

Both backends run one behavioural suite (`crates/chapkit-store/tests/contract.rs`), and
`make test-turso` also runs the full HTTP contract, the chap-core flow, on turso.

**Schema:** `configs`, `artifacts` (tree via `parent_id` and `level`, JSON `data`, optional `content`
blob kept out of the JSON), `artifact_listings` and `config_artifacts`. Deleting a config deletes the
artifact trees linked to it, as chapkit does.

**Listings:** every artifact write also writes its listing row: `data` without `content`, plus a small
`summary` for backtests and forecasts (`chapkit_store::summary`), with `data.type` and
`data.content.dataset_id` as columns. Listing artifacts, filtering them by type and dataset (`q`) and
counting them read only that table, never an artifact's `data`, so a page of results costs the same
however large the results are. A database from before listings gets them when it is opened, one
artifact at a time.

**Datasets** are stored as their chapkit JSON. The service keeps parsed datasets in an LRU cache of at
most 1 GiB (estimated from rows and column types), since datasets never change once stored: `$data`
projections, backtests and forecasts of a cached dataset do not parse it again. A dataset larger than
the whole cache is parsed per request.
