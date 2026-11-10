# Finishing fast jobs before chap-core polls

**Decision:** `$train` and `$predict` submit the job, then wait up to a budget (default 1 s,
`CHAPKIT_EAGER_WAIT_MS`) for it to finish before replying 202. The reply is unchanged.

**Why:** chap-core's `wait_for_job` asks for the job status first and sleeps 2 seconds only if the job is
not finished. A job that is already finished when the first status request arrives costs no sleep at all.
Python chapkit's jobs always take longer than the time between the 202 and that first request, so every
call pays at least one 2-second sleep. A Rust job that finishes in microseconds would usually win that race
anyway, but "usually" is not a guarantee; waiting for it before replying makes it deterministic.

**What it costs:** nothing for fast jobs. A job that takes longer than the budget replies after the budget
and behaves exactly as before: chap-core polls and sleeps. The budget only bounds how long the HTTP reply is
delayed.

**Not specific to Rust:** Python chapkit could do the same. Most of its jobs would still exceed a short
budget because of the workspace copy and pickling, but the change is cheap and would help its fastest
models. The benchmarks report this effect separately from the speed of Rust, by running chapkit-rs with
the budget set to zero.
