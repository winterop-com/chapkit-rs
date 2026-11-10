//! `chapkit-rs`: scaffold a Rust model package (`init`) and run chap-core's flow against a running
//! service (`test`).

mod flow;
mod init;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "chapkit-rs",
    version,
    about = "Build and test chapkit-rs model packages"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a new model package: the calculation in src/model.rs, the chapkit plumbing in
    /// src/lib.rs, the service in src/main.rs, a test, a Dockerfile and a compose file.
    Init {
        /// Package name, also the service id chap-core sees: lowercase words joined by hyphens.
        name: String,
        /// Directory to create (default: the name).
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Depend on a local chapkit-rs checkout instead of the git repository.
        #[arg(long, value_name = "PATH")]
        chapkit_path: Option<PathBuf>,
        /// Display name (default: from the package name).
        #[arg(long)]
        title: Option<String>,
    },
    /// Run chap-core's flow against a running service: health, info, configuration, sample data,
    /// training and forecasting jobs, and the shape of the stored forecast.
    Test {
        /// Base URL of the service.
        #[arg(long, default_value = "http://127.0.0.1:8000")]
        url: String,
        /// Locations in the generated data.
        #[arg(long, default_value_t = 5)]
        locations: usize,
        /// Periods of training data.
        #[arg(long, default_value_t = 36)]
        periods: usize,
        /// Seconds to wait for each job.
        #[arg(long, default_value_t = 300)]
        timeout: u64,
        /// Keep the configuration and artifacts the test creates.
        #[arg(long)]
        keep: bool,
    },
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
        Command::Test {
            url,
            locations,
            periods,
            timeout,
            keep,
        } => {
            flow::run(flow::Options {
                url,
                locations,
                periods,
                timeout,
                keep,
            })
            .await
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
