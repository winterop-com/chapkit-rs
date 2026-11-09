//! Self-registration with a Chap orchestrator (chap-core `/v2/services/$register`),
//! following servicekit: register, keep alive with `PUT ping_url`, re-register
//! after a 404, deregister on shutdown.

use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::watch;

/// Registration settings, read from servicekit's environment variables.
#[derive(Debug, Clone)]
pub struct Registration {
    /// `SERVICEKIT_ORCHESTRATOR_URL`, e.g. `http://chap:8000/v2/services/$register`.
    pub orchestrator_url: String,
    /// The URL others reach this service at.
    pub service_url: String,
    /// `SERVICEKIT_REGISTRATION_KEY`, sent as `X-Service-Key`.
    pub service_key: Option<String>,
    /// Keepalive interval.
    pub keepalive: Duration,
    /// Wait after a 404 ping before registering again.
    pub re_register_grace: Duration,
    /// Attempts for the first registration.
    pub max_retries: u32,
    /// Pause between attempts.
    pub retry_delay: Duration,
}

impl Registration {
    /// Settings from the environment; `None` without `SERVICEKIT_ORCHESTRATOR_URL`.
    /// The advertised URL is `http://{SERVICEKIT_HOST|hostname}:{SERVICEKIT_PORT|PORT|8000}`.
    pub fn from_env() -> Option<Self> {
        let orchestrator_url = std::env::var("SERVICEKIT_ORCHESTRATOR_URL")
            .ok()
            .filter(|s| !s.is_empty())?;
        let host = std::env::var("SERVICEKIT_HOST")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| gethostname::gethostname().to_string_lossy().into_owned());
        let port = std::env::var("SERVICEKIT_PORT")
            .or_else(|_| std::env::var("PORT"))
            .unwrap_or_else(|_| "8000".into());
        Some(Self {
            orchestrator_url,
            service_url: format!("http://{host}:{port}"),
            service_key: std::env::var("SERVICEKIT_REGISTRATION_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            keepalive: Duration::from_secs(10),
            re_register_grace: Duration::from_secs(30),
            max_retries: 5,
            retry_delay: Duration::from_secs(2),
        })
    }

    fn deregister_url(&self, id: &str) -> String {
        let base = self
            .orchestrator_url
            .trim_end_matches('/')
            .trim_end_matches("/$register");
        format!("{base}/{id}")
    }
}

#[derive(Deserialize)]
struct Registered {
    ping_url: String,
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap_or_default()
}

async fn register(
    c: &reqwest::Client,
    cfg: &Registration,
    id: &str,
    info: &Value,
) -> Result<String, String> {
    let mut req = c
        .post(&cfg.orchestrator_url)
        .json(&json!({"id": id, "url": cfg.service_url, "info": info}));
    if let Some(k) = &cfg.service_key {
        req = req.header("X-Service-Key", k);
    }
    let res = req.send().await.map_err(|e| e.to_string())?;
    if !res.status().is_success() {
        return Err(format!("orchestrator returned {}", res.status()));
    }
    res.json::<Registered>()
        .await
        .map(|r| r.ping_url)
        .map_err(|e| e.to_string())
}

/// Run registration until `shutdown` flips, then deregister. Spawn this on the runtime.
pub async fn run(cfg: Registration, id: String, info: Value, mut shutdown: watch::Receiver<bool>) {
    let c = client();
    let mut ping_url = None;
    for attempt in 1..=cfg.max_retries {
        match register(&c, &cfg, &id, &info).await {
            Ok(url) => {
                tracing::info!(%id, url = %cfg.service_url, "registration.success");
                ping_url = Some(url);
                break;
            }
            Err(e) => tracing::warn!(attempt, error = %e, "registration.failed"),
        }
        tokio::select! {
            () = tokio::time::sleep(cfg.retry_delay) => {}
            _ = shutdown.changed() => return,
        }
    }
    let Some(mut ping_url) = ping_url else {
        tracing::error!("registration.gave_up");
        return;
    };

    loop {
        tokio::select! {
            () = tokio::time::sleep(cfg.keepalive) => {}
            _ = shutdown.changed() => break,
        }
        let mut req = c.put(&ping_url);
        if let Some(k) = &cfg.service_key {
            req = req.header("X-Service-Key", k);
        }
        match req.send().await {
            Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => {
                tracing::warn!("registration.keepalive_not_found");
                tokio::time::sleep(cfg.re_register_grace).await;
                if let Ok(url) = register(&c, &cfg, &id, &info).await {
                    ping_url = url;
                }
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(error = %e, "registration.keepalive_failed"),
        }
    }

    let mut req = c.delete(cfg.deregister_url(&id));
    if let Some(k) = &cfg.service_key {
        req = req.header("X-Service-Key", k);
    }
    match req.send().await {
        Ok(_) => tracing::info!(%id, "registration.deregistered"),
        Err(e) => tracing::warn!(error = %e, "registration.deregister_failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deregister_url_strips_register_suffix() {
        let cfg = Registration {
            orchestrator_url: "http://chap:8000/v2/services/$register".into(),
            service_url: String::new(),
            service_key: None,
            keepalive: Duration::from_secs(1),
            re_register_grace: Duration::from_secs(1),
            max_retries: 1,
            retry_delay: Duration::from_secs(1),
        };
        assert_eq!(cfg.deregister_url("x"), "http://chap:8000/v2/services/x");
    }
}
