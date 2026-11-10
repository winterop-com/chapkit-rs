# Install the tools

By the end of this page you have Rust and the `chapkit-rs` command.

## Before you start

- macOS or Linux (on Windows, use WSL).

**What you do not need:** Python, chap-core or Docker. A model package is a Rust program; chap-core and Docker
come in when you [connect it to chap-core](../guide/chap-core.md) or [ship it as a container](ship.md).

## 1. Install Rust

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Open a new terminal, then:

```sh
cargo --version        # cargo 1.99 or newer
```

## 2. Install the chapkit-rs command

Each release on [GitHub](https://github.com/winterop-com/chapkit-rs/releases) has the command for Linux (static,
amd64 and arm64) and macOS (Apple Silicon and Intel). For example, on an Apple Silicon Mac:

```sh
curl -sL https://github.com/winterop-com/chapkit-rs/releases/latest/download/chapkit-rs-aarch64-apple-darwin.tar.gz | tar xz
install chapkit-rs-*/chapkit-rs ~/.local/bin/
chapkit-rs --version
```

The other archives are `chapkit-rs-x86_64-apple-darwin`, `chapkit-rs-x86_64-unknown-linux-musl` and
`chapkit-rs-aarch64-unknown-linux-musl`; `SHA256SUMS` beside them checks a download. To build a release from source
instead (into `~/.cargo/bin`):

```sh
cargo install --git https://github.com/winterop-com/chapkit-rs --tag v0.2.0 chapkit-cli
```

## If something goes wrong

- **`chapkit-rs: command not found`**: `~/.cargo/bin` is not on your `PATH`; run `source ~/.cargo/env` or open
  a new terminal.

## Next steps

- [Your first model package](first-model.md)
