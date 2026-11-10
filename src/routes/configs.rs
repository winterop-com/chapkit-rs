//! `/api/v1/configs`.

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use chapkit_http::time::entity_now;
use chapkit_http::{ApiResult, Json, ListOrPage, PageParams, Problem, Query, Stats, parse_ulid};
use chapkit_store::{ConfigRow, Page};
use serde_json::Value;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use crate::erased::without_content;
use crate::state::{AppState, store_problem, validate_tags};
use crate::wire::{ArtifactOut, ConfigIn, ConfigOut, LinkArtifact};

const PREFIX: &str = "/api/v1/configs";

fn not_found(id: &str) -> Problem {
    Problem::not_found(format!("Config with id {id} not found")).instance(format!("{PREFIX}/{id}"))
}

fn normalized(state: &AppState, body: &ConfigIn) -> ApiResult<Value> {
    validate_tags(&body.tags)?;
    state
        .model()
        .normalize_config(body.data.clone())
        .map_err(|msg| {
            Problem::validation(vec![
                serde_json::json!({"loc": ["body", "data"], "msg": msg, "type": "value_error"}),
            ])
        })
}

/// List configs (paged when both `page` and `size` are given).
#[utoipa::path(get, path = "", tag = "Config", params(PageParams),
    responses((status = OK, body = ListOrPage<ConfigOut>), (status = UNPROCESSABLE_ENTITY, body = Problem)))]
async fn list(
    State(s): State<AppState>,
    Query(p): Query<PageParams>,
) -> ApiResult<ListOrPage<ConfigOut>> {
    let paging = p.resolve()?;
    let page = paging.map(|(page, size)| Page {
        offset: chapkit_http::extract::page_offset(page, size),
        limit: size,
    });
    let (rows, total) = s.store().list_configs(page).await.map_err(store_problem)?;
    let items: Vec<ConfigOut> = rows.into_iter().map(Into::into).collect();
    Ok(ListOrPage::new(items, total, paging))
}

/// Create a config. Defaults are filled; keys you send come back unchanged.
#[utoipa::path(post, path = "", tag = "Config", request_body = ConfigIn,
    responses((status = CREATED, body = ConfigOut), (status = CONFLICT, body = Problem), (status = UNPROCESSABLE_ENTITY, body = Problem)))]
async fn create(
    State(s): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ConfigIn>,
) -> ApiResult<Response> {
    let data = normalized(&s, &body)?;
    let id = match &body.id {
        Some(raw) => parse_ulid(raw)?.to_string(),
        None => ulid::Ulid::generate().to_string(),
    };
    let now = entity_now();
    let row = ConfigRow {
        id: id.clone(),
        name: body.name,
        data,
        tags: body.tags,
        created_at: now.clone(),
        updated_at: now,
    };
    s.store()
        .insert_config(row.clone())
        .await
        .map_err(store_problem)?;
    let host = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost");
    let mut res = (StatusCode::CREATED, axum::Json(ConfigOut::from(row))).into_response();
    if let Ok(v) = format!("http://{host}{PREFIX}/{id}").parse() {
        res.headers_mut().insert(header::LOCATION, v);
    }
    Ok(res)
}

/// Get a config.
#[utoipa::path(get, path = "/{id}", tag = "Config", params(("id" = String, Path)),
    responses((status = OK, body = ConfigOut), (status = NOT_FOUND, body = Problem), (status = BAD_REQUEST, body = Problem)))]
async fn get(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<axum::Json<ConfigOut>> {
    parse_ulid(&id)?;
    s.store()
        .get_config(&id)
        .await
        .map_err(store_problem)?
        .map(|r| axum::Json(r.into()))
        .ok_or_else(|| not_found(&id))
}

/// Replace a config.
#[utoipa::path(put, path = "/{id}", tag = "Config", params(("id" = String, Path)), request_body = ConfigIn,
    responses((status = OK, body = ConfigOut), (status = NOT_FOUND, body = Problem)))]
async fn update(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ConfigIn>,
) -> ApiResult<axum::Json<ConfigOut>> {
    parse_ulid(&id)?;
    let data = normalized(&s, &body)?;
    let existing = s
        .store()
        .get_config(&id)
        .await
        .map_err(store_problem)?
        .ok_or_else(|| not_found(&id))?;
    let row = ConfigRow {
        name: body.name,
        data,
        tags: body.tags,
        updated_at: entity_now(),
        ..existing
    };
    s.store()
        .update_config(row)
        .await
        .map_err(store_problem)?
        .map(|r| axum::Json(r.into()))
        .ok_or_else(|| not_found(&id))
}

/// Delete a config and the artifact trees linked to it.
#[utoipa::path(delete, path = "/{id}", tag = "Config", params(("id" = String, Path)),
    responses((status = NO_CONTENT), (status = NOT_FOUND, body = Problem)))]
async fn delete(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<StatusCode> {
    parse_ulid(&id)?;
    if s.store().delete_config(&id).await.map_err(store_problem)? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(not_found(&id))
    }
}

/// JSON schema of the config `data` (model options plus reserved keys).
#[utoipa::path(get, path = "/$schema", tag = "Config", responses((status = OK, body = Object)))]
async fn schema(State(s): State<AppState>) -> axum::Json<Value> {
    axum::Json(s.model().config_schema())
}

/// Config count.
#[utoipa::path(get, path = "/$stats", tag = "Config", responses((status = OK, body = Stats)))]
async fn stats(State(s): State<AppState>) -> ApiResult<axum::Json<Stats>> {
    let (_, total) = s
        .store()
        .list_configs(Some(Page {
            offset: 0,
            limit: 0,
        }))
        .await
        .map_err(store_problem)?;
    Ok(axum::Json(Stats { total }))
}

/// Link a root artifact to the config.
#[utoipa::path(post, path = "/{id}/$link-artifact", tag = "Config", params(("id" = String, Path)), request_body = LinkArtifact,
    responses((status = NO_CONTENT), (status = BAD_REQUEST, body = Problem), (status = NOT_FOUND, body = Problem)))]
async fn link(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(b): Json<LinkArtifact>,
) -> ApiResult<StatusCode> {
    parse_ulid(&id)?;
    s.store()
        .link(&id, &b.artifact_id)
        .await
        .map_err(store_problem)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Unlink an artifact from the config.
#[utoipa::path(post, path = "/{id}/$unlink-artifact", tag = "Config", params(("id" = String, Path)), request_body = LinkArtifact,
    responses((status = NO_CONTENT), (status = NOT_FOUND, body = Problem)))]
async fn unlink(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(b): Json<LinkArtifact>,
) -> ApiResult<StatusCode> {
    parse_ulid(&id)?;
    s.store()
        .unlink(&id, &b.artifact_id)
        .await
        .map_err(store_problem)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Root artifacts linked to the config.
#[utoipa::path(get, path = "/{id}/$artifacts", tag = "Config", params(("id" = String, Path)),
    responses((status = OK, body = Vec<ArtifactOut>), (status = NOT_FOUND, body = Problem)))]
async fn artifacts(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<axum::Json<Vec<ArtifactOut>>> {
    parse_ulid(&id)?;
    if s.store()
        .get_config(&id)
        .await
        .map_err(store_problem)?
        .is_none()
    {
        return Err(not_found(&id));
    }
    let rows = s
        .store()
        .linked_artifacts(&id)
        .await
        .map_err(store_problem)?;
    Ok(axum::Json(
        rows.into_iter()
            .map(|r| ArtifactOut {
                data: without_content(&r.data),
                ..r.into()
            })
            .collect(),
    ))
}

/// The router (mount at `/api/v1/configs`).
pub fn router(state: AppState) -> OpenApiRouter {
    OpenApiRouter::new()
        .routes(routes!(list, create))
        .routes(routes!(schema))
        .routes(routes!(stats))
        .routes(routes!(get, update, delete))
        .routes(routes!(link))
        .routes(routes!(unlink))
        .routes(routes!(artifacts))
        .with_state(state)
}
