//! The app builder: assembles routers, docs, middleware, and serves.

use std::sync::Arc;

use axum::Router;
use axum::response::IntoResponse;
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::watch;
use utoipa::openapi::{InfoBuilder, OpenApi};
use utoipa_axum::router::OpenApiRouter;
use utoipa_scalar::{Scalar, Servable};

use crate::health::{self, Check, Checks};
use crate::middleware::{ApiKeys, request_id, require_api_key};
use crate::problem::Problem;
use crate::registration::{self, Registration};
use crate::system;

/// Builds an HTTP service with servicekit's standard endpoints.
pub struct App {
    id: String,
    title: String,
    version: String,
    info: Value,
    api: OpenApiRouter,
    checks: Checks,
    auth: Option<ApiKeys>,
    registration: Option<Registration>,
}

impl App {
    /// A new app. `info` is served at `/api/v1/info` and sent on registration.
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        version: impl Into<String>,
        info: Value,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            version: version.into(),
            info,
            api: OpenApiRouter::new(),
            checks: Checks::default(),
            auth: ApiKeys::from_env(),
            registration: Registration::from_env(),
        }
    }

    /// Mount a router under `prefix`.
    #[must_use]
    pub fn nest(mut self, prefix: &str, router: OpenApiRouter) -> Self {
        self.api = self.api.nest(prefix, router);
        self
    }

    /// Merge a router at the root.
    #[must_use]
    pub fn merge(mut self, router: OpenApiRouter) -> Self {
        self.api = self.api.merge(router);
        self
    }

    /// Add a health check.
    #[must_use]
    pub fn health_check(mut self, name: impl Into<String>, check: Check) -> Self {
        self.checks.0.push((name.into(), check));
        self
    }

    /// Require API keys (overrides `SERVICEKIT_API_KEYS`).
    #[must_use]
    pub fn auth(mut self, keys: ApiKeys) -> Self {
        self.auth = Some(keys);
        self
    }

    /// Register with an orchestrator (overrides `SERVICEKIT_ORCHESTRATOR_URL`).
    #[must_use]
    pub fn registration(mut self, r: Option<Registration>) -> Self {
        self.registration = r;
        self
    }

    /// The OpenAPI document and the axum router.
    pub fn build(self) -> (Router, OpenApi, Option<Registration>, String, Value) {
        let (router, mut doc) = self
            .api
            .merge(health::router(self.checks))
            .merge(crate::metrics::router())
            .merge(system::router(self.info.clone()))
            .split_for_parts();
        doc.info = InfoBuilder::new()
            .title(self.title)
            .version(self.version)
            .build();
        let doc_json = doc.clone();
        let mut router = router
            .route(
                "/openapi.json",
                axum::routing::get(move || async move { axum::Json(doc_json) }),
            )
            .merge(Scalar::with_url("/docs", doc.clone()))
            .fallback(|uri: axum::http::Uri| async move {
                Problem::not_found(format!("No route for {}", uri.path()))
                    .instance(uri.path().to_owned())
                    .into_response()
            });
        if let Some(keys) = self.auth {
            router = router.layer(axum::middleware::from_fn_with_state(
                Arc::new(keys),
                require_api_key,
            ));
        }
        let router = router.layer(axum::middleware::from_fn(request_id));
        // Decode `%24` to `$` before routing, so `$`-operations match however they are
        // spelled. Starlette (Python chapkit) routes on the decoded path and chap-core's
        // service proxy forwards `%24`; axum routes on the raw path.
        let router = Router::new().fallback_service(
            tower::ServiceBuilder::new()
                .map_request(decode_dollar)
                .service(router),
        );
        (router, doc, self.registration, self.id, self.info)
    }

    /// Serve on `HOST:PORT` (defaults `127.0.0.1:8000`) until Ctrl-C / SIGTERM.
    pub async fn serve(self) -> std::io::Result<()> {
        init_tracing();
        let (router, _, registration, id, info) = self.build();
        let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
        let port = std::env::var("PORT").unwrap_or_else(|_| "8000".into());
        let listener = TcpListener::bind(format!("{host}:{port}")).await?;
        // chap-core's MLproject runner waits for uvicorn's startup line; print the same.
        println!("INFO:     Uvicorn running on http://{host}:{port} (Press CTRL+C to quit)");
        tracing::info!(%host, %port, "service.start");

        let (tx, rx) = watch::channel(false);
        let reg = registration.map(|r| tokio::spawn(registration::run(r, id, info, rx)));
        let res = axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal())
            .await;
        let _ = tx.send(true);
        if let Some(h) = reg {
            let _ = h.await;
        }
        res
    }
}

/// Rewrite `%24` (any case) in the request path to `$`; the query is left untouched.
fn decode_dollar<B>(mut req: axum::http::Request<B>) -> axum::http::Request<B> {
    let uri = req.uri();
    let path = uri.path();
    if !path.contains("%24") {
        return req;
    }
    let decoded = path.replace("%24", "$");
    let pq = match uri.query() {
        Some(q) => format!("{decoded}?{q}"),
        None => decoded,
    };
    let mut parts = uri.clone().into_parts();
    if let Ok(v) = pq.parse() {
        parts.path_and_query = Some(v);
        if let Ok(new_uri) = axum::http::Uri::from_parts(parts) {
            *req.uri_mut() = new_uri;
        }
    }
    req
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = term => {}
    }
}

/// Configure tracing from `LOG_LEVEL` (default `info`) and `LOG_FORMAT` (`console` or `json`).
pub fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_new(
        std::env::var("LOG_LEVEL")
            .unwrap_or_else(|_| "info".into())
            .to_lowercase(),
    )
    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    let _ = if std::env::var("LOG_FORMAT").is_ok_and(|f| f == "json") {
        builder.json().try_init()
    } else {
        builder.try_init()
    };
}
