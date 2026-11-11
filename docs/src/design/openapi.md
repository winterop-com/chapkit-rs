# OpenAPI as a reviewed artifact

The OpenAPI document is generated from the code with utoipa: every handler carries its path, parameters,
bodies and responses. Three things keep it honest:

1. **Committed.** `openapi.json` is in the repository and `tests/openapi.rs` fails when the generated
   document differs from it. Any API change therefore shows up in review as a diff of that file.
   `make openapi` regenerates it.
2. **Compared with Python chapkit.** `make compare-openapi PY=<url> RS=<url>` (`scripts/compare_openapi.py`)
   diffs a running Python chapkit service against a chapkit-rs one: routes only one side has, success status
   codes, and request and response body fields. It exits non-zero when a route chap-core relies on is missing
   or differs.
3. **Served.** Every service serves `/openapi.json` and interactive docs at `/docs` (Scalar).

Current state against Python chapkit 2.6: every one of its 38 operations exists in chapkit-rs, plus
`$predict-sync`, `$predict-splits`, `$explain`, `$backtest` and `$hpo`. The remaining differences are cosmetic: `/api/v1/system` has an extra
`chapkit_rs_version` field, `/api/v1/system/apps` returns an empty list (no bundled web apps), and
`$validate` documents one flat request body where Python documents two variants.
