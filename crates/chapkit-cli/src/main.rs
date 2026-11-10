//! `chapkit-rs`: scaffold a Rust model package (`init`) and run chap-core's flow against a running
//! service (`test`).

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
    /// src/main.rs; a test, a Dockerfile and a compose file come with it.
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
    /// Runs chap-core's flow against it: health, info, configuration, sample data, training and
    /// forecasting jobs, and the shape of the stored forecast.
    Test {
        /// Service URL
        #[arg(long, default_value = "http://127.0.0.1:8000")]
        url: String,
        /// Locations in the test data
        #[arg(long, default_value_t = 5)]
        locations: usize,
        /// Training periods
        #[arg(long, default_value_t = 36)]
        periods: usize,
        /// Seconds to wait per job
        #[arg(long, default_value_t = 300)]
        timeout: u64,
        /// Keep what the test creates
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
