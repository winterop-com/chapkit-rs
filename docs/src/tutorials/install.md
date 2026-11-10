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

```sh
cargo install --git https://github.com/winterop-com/chapkit-rs chapkit-cli
chapkit-rs --version
```

This builds the command from source and puts it in `~/.cargo/bin`.

## If something goes wrong

- **`chapkit-rs: command not found`**: `~/.cargo/bin` is not on your `PATH`; run `source ~/.cargo/env` or open
  a new terminal.

## Next steps

- [Your first model package](first-model.md)
