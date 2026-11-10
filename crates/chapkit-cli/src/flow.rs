//! `chapkit-rs test`: chap-core's flow (`chapkit_flow`) over HTTP against a running chapkit service,
//! Rust or Python.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

/// `chapkit-rs test` options.
#[derive(Debug, clap::Args)]
pub struct Options {
    /// Service URL
    #[arg(long, short, default_value = "http://127.0.0.1:8000")]
    url: String,
    #[command(flatten)]
    flow: chapkit_flow::Options,
    /// Start the package here (cargo run) first
    #[arg(long)]
    start_service: bool,
}

/// Run the test; `Err` when a required step or any job failed.
pub async fn run(o: Options) -> Result<(), String> {
    // Kept alive until the test is done; dropping it stops the service.
    let _service = if o.start_service {
        Some(start_service(&o.url, o.flow.timeout).await?)
    } else {
        None
    };
    chapkit_flow::run(&o.url, &o.flow).await
}

/// Start the package in the current directory with `cargo run --release -- serve` on the URL's port and
/// wait until it is healthy. The service stops when the returned child is dropped.
async fn start_service(url: &str, timeout: u64) -> Result<tokio::process::Child, String> {
    let dir = find_package(&std::env::current_dir().map_err(|e| e.to_string())?)
        .ok_or("--start-service: no Cargo.toml with a [package] here or above")?;
    let port = reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.port_or_known_default())
        .ok_or_else(|| format!("--start-service: no port in {url}"))?;
    println!(
        "Starting {} on port {port} (cargo run --release -- serve)...",
        dir.display()
    );
    let mut child = tokio::process::Command::new("cargo")
        .args(["run", "--release", "--quiet", "--", "serve"])
        .current_dir(&dir)
        .env("HOST", "127.0.0.1")
        .env("PORT", port.to_string())
        .env("DATABASE_URL", ":memory:")
        .env("LOG_LEVEL", "error")
        .stdout(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("cargo run: {e}"))?;
    // Building can take minutes the first time.
    let deadline = Instant::now() + Duration::from_secs(timeout.max(600));
    let health = format!("{}/health", url.trim_end_matches('/'));
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("the service exited ({status}) before it was ready"));
        }
        if reqwest::get(&health)
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            println!("Service ready\n");
            return Ok(child);
        }
        if Instant::now() > deadline {
            return Err("the service did not become ready in time".into());
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn find_package(start: &Path) -> Option<PathBuf> {
    start.ancestors().find_map(|dir| {
        let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).ok()?;
        manifest.contains("[package]").then(|| dir.to_path_buf())
    })
}
