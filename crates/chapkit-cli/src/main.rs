//! `chapkit-rs`: scaffold a Rust model package (`init`), test a running service (`test`), and
//! inspect what a service stored (`artifact`).

mod artifact;
mod flow;
mod init;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "chapkit-rs", version, about = "chapkit-rs model packages")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a model package
    ///
    /// The calculation goes in src/model.rs, the chapkit plumbing in src/lib.rs and the service in
    /// src/main.rs; a test, a Dockerfile, a compose file and a GitHub workflow that publishes the
    /// image come with it.
    Init {
        /// Package and service id, e.g. my-model
        name: String,
        /// Directory to create [default: NAME]
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Use a local chapkit-rs checkout
        #[arg(long, value_name = "PATH")]
        chapkit_path: Option<PathBuf>,
        /// Display name
        #[arg(long)]
        title: Option<String>,
    },
    /// Test a running service
    ///
    /// Runs chap-core's flow against it with generated data shaped by what the service declares:
    /// configurations from its schema, $validate, training and forecasting jobs, and the shape of
    /// every stored forecast. Works against Python chapkit services too.
    Test(flow::Options),
    /// Inspect stored artifacts
    #[command(subcommand)]
    Artifact(artifact::Command),
}

#[tokio::main]
async fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Init {
            name,
            dir,
            chapkit_path,
            title,
        } => init::run(&name, dir, chapkit_path.as_deref(), title),
        Command::Test(options) => flow::run(options).await,
        Command::Artifact(command) => artifact::run(command).await,
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
