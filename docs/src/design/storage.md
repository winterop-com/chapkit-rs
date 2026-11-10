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
blob kept out of the JSON) and `config_artifacts`. Deleting a config deletes the artifact trees linked to
it, as chapkit does.
