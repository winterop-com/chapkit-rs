# Errors and panics

**Library code does not panic on input.** `unwrap()`, `expect()`, `panic!`, `todo!` and `unimplemented!`
are denied by clippy in library and example code (`[workspace.lints.clippy]`); tests may use them, because a
panic is how a test fails. Indexing into caller-supplied data uses `get()` and returns an error.

**Errors are layered.** `chapkit-core::Error` knows nothing about HTTP (`InvalidInput`, `Config`, `Model`,
`Frame`, `Serde`). The service maps errors to responses at the edge:

- request problems become RFC 9457 `application/problem+json` responses with servicekit's error types
  (`urn:servicekit:error:not-found`, `invalid-ulid`, `validation-failed`, `conflict`, `unauthorized`,
  `bad-request`, `internal`, `database`), titles and status codes;
- failures inside a job become a failed job record with `error` as `"Kind: message"` (for example
  `ValueError: Config 01J... not found`), as Python chapkit reports them, which chap-core shows to users;
- database errors are logged and returned as a generic message; SQL is never sent to clients.
