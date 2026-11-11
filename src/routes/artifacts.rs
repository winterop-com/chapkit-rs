//! `/api/v1/artifacts`.

use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use chapkit_http::time::entity_now;
use chapkit_http::{ApiResult, Json, ListOrPage, PageParams, Problem, Query, Stats, parse_ulid};
use chapkit_store::{NewArtifact, Page};
use serde_json::Value;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use crate::erased::without_content;
use crate::ml::{HIERARCHY, level_label};
use crate::state::{AppState, store_problem, str_field, validate_tags};
use crate::wire::{ArtifactIn, ArtifactOut, ArtifactTreeNode, ConfigOut};

const PREFIX: &str = "/api/v1/artifacts";

fn not_found(id: &str) -> Problem {
    Problem::not_found(format!("Artifact with id {id} not found"))
        .instance(format!("{PREFIX}/{id}"))
}

async fn fetch(s: &AppState, id: &str) -> ApiResult<ArtifactOut> {
    parse_ulid(id)?;
    s.store()
        .get_artifact(id)
        .await
        .map_err(store_problem)?
        .map(Into::into)
        .ok_or_else(|| not_found(id))
}

fn node(a: ArtifactOut, children: Option<Vec<ArtifactTreeNode>>) -> ArtifactTreeNode {
    // Analysis results are roots of their own kind, not trainings.
    let label = match str_field(&a.data, "type") {
        Some(
            t @ (crate::analysis::BACKTEST_TYPE
            | crate::analysis::HPO_TYPE
            | crate::analysis::FORECAST_TYPE),
        ) => t.to_owned(),
        _ => level_label(a.level),
    };
    ArtifactTreeNode {
        artifact: ArtifactOut {
            data: without_content(&a.data),
            ..a
        },
        level_label: Some(label),
        hierarchy: Some(HIERARCHY.into()),
        children,
    }
}

/// `page` and `size`, plus the chapkit-rs filter `type`.
#[derive(Debug, Clone, Default, serde::Deserialize, utoipa::IntoParams)]
struct ListParams {
    /// Page number, starting at 1.
    page: Option<i64>,
    /// Page size, 1-100.
    size: Option<i64>,
    /// Only artifacts whose `data.type` is this (`ml_training`, `ml_prediction`, `ml_backtest`,
    /// `ml_forecast`, ...). chapkit-rs extension.
    #[serde(rename = "type")]
    kind: Option<String>,
}

/// List artifact summaries (`data` without `content`), optionally of one `type`.
#[utoipa::path(get, path = "", tag = "Artifacts", params(ListParams),
    responses((status = OK, body = ListOrPage<ArtifactOut>), (status = UNPROCESSABLE_ENTITY, body = Problem)))]
async fn list(
    State(s): State<AppState>,
    Query(p): Query<ListParams>,
) -> ApiResult<ListOrPage<ArtifactOut>> {
    let paging = PageParams {
        page: p.page,
        size: p.size,
    }
    .resolve()?;
    let page = paging.map(|(page, size)| Page {
        offset: chapkit_http::extract::page_offset(page, size),
        limit: size,
    });
    let (rows, total) = match p.kind.as_deref() {
        Some(kind) => s.store().list_artifacts_of_type(kind, page).await,
        None => s.store().list_artifacts(page).await,
    }
    .map_err(store_problem)?;
    let items: Vec<ArtifactOut> = rows
        .into_iter()
        .map(|r| ArtifactOut {
            data: without_content(&r.data),
            ..r.into()
        })
        .collect();
    Ok(ListOrPage::new(items, total, paging))
}

/// Create an artifact. The level is the parent's level + 1.
#[utoipa::path(post, path = "", tag = "Artifacts", request_body = ArtifactIn,
    responses((status = CREATED, body = ArtifactOut), (status = CONFLICT, body = Problem)))]
async fn create(
    State(s): State<AppState>,
    Json(b): Json<ArtifactIn>,
) -> ApiResult<(StatusCode, axum::Json<ArtifactOut>)> {
    validate_tags(&b.tags)?;
    let id = match &b.id {
        Some(raw) => parse_ulid(raw)?.to_string(),
        None => chapkit_http::ids::new_id().to_string(),
    };
    if let Some(p) = &b.parent_id {
        parse_ulid(p)?;
    }
    let row = s
        .store()
        .insert_artifact(NewArtifact {
            id,
            parent_id: b.parent_id,
            data: b.data,
            tags: b.tags,
            content: None,
            created_at: entity_now(),
        })
        .await
        .map_err(store_problem)?;
    Ok((StatusCode::CREATED, axum::Json(row.into())))
}

/// Get an artifact.
#[utoipa::path(get, path = "/{id}", tag = "Artifacts", params(("id" = String, Path)),
    responses((status = OK, body = ArtifactOut), (status = NOT_FOUND, body = Problem)))]
async fn get(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<axum::Json<ArtifactOut>> {
    fetch(&s, &id).await.map(axum::Json)
}

/// Replace an artifact's data and tags.
#[utoipa::path(put, path = "/{id}", tag = "Artifacts", params(("id" = String, Path)), request_body = ArtifactIn,
    responses((status = OK, body = ArtifactOut), (status = NOT_FOUND, body = Problem)))]
async fn update(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(b): Json<ArtifactIn>,
) -> ApiResult<axum::Json<ArtifactOut>> {
    parse_ulid(&id)?;
    validate_tags(&b.tags)?;
    s.store()
        .update_artifact(&id, b.data, b.tags, entity_now())
        .await
        .map_err(store_problem)?
        .map(|r| axum::Json(r.into()))
        .ok_or_else(|| not_found(&id))
}

/// Delete an artifact.
#[utoipa::path(delete, path = "/{id}", tag = "Artifacts", params(("id" = String, Path)),
    responses((status = NO_CONTENT), (status = NOT_FOUND, body = Problem)))]
async fn delete(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<StatusCode> {
    parse_ulid(&id)?;
    if s.store()
        .delete_artifact(&id)
        .await
        .map_err(store_problem)?
    {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(not_found(&id))
    }
}

/// JSON schema of an artifact.
#[utoipa::path(get, path = "/$schema", tag = "Artifacts", responses((status = OK, body = Object)))]
async fn schema() -> axum::Json<Value> {
    axum::Json(serde_json::json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"}, "created_at": {"type": "string"}, "updated_at": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}, "data": {},
            "parent_id": {"type": ["string", "null"]}, "level": {"type": "integer"}
        },
        "required": ["id", "created_at", "updated_at", "data", "level"]
    }))
}

/// Artifact count.
#[utoipa::path(get, path = "/$stats", tag = "Artifacts", responses((status = OK, body = Stats)))]
async fn stats(State(s): State<AppState>) -> ApiResult<axum::Json<Stats>> {
    let (_, total) = s
        .store()
        .list_artifacts(Some(Page {
            offset: 0,
            limit: 0,
        }))
        .await
        .map_err(store_problem)?;
    Ok(axum::Json(Stats { total }))
}

/// The artifact as a tree node, without children.
#[utoipa::path(get, path = "/{id}/$expand", tag = "Artifacts", params(("id" = String, Path)),
    responses((status = OK, body = ArtifactTreeNode), (status = NOT_FOUND, body = Problem)))]
async fn expand(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<axum::Json<ArtifactTreeNode>> {
    Ok(axum::Json(node(fetch(&s, &id).await?, None)))
}

/// The artifact's whole subtree (content stripped).
#[utoipa::path(get, path = "/{id}/$tree", tag = "Artifacts", params(("id" = String, Path)),
    responses((status = OK, body = ArtifactTreeNode), (status = NOT_FOUND, body = Problem)))]
async fn tree(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<axum::Json<ArtifactTreeNode>> {
    let root = fetch(&s, &id).await?;
    Ok(axum::Json(Box::pin(subtree(&s, root, 0)).await?))
}

async fn subtree(s: &AppState, a: ArtifactOut, depth: usize) -> ApiResult<ArtifactTreeNode> {
    let kids = if depth >= 64 {
        vec![]
    } else {
        s.store().children(&a.id).await.map_err(store_problem)?
    };
    let mut children = Vec::with_capacity(kids.len());
    for k in kids {
        children.push(Box::pin(subtree(s, k.into(), depth + 1)).await?);
    }
    Ok(node(a, Some(children)))
}

/// The config linked to the artifact's root.
#[utoipa::path(get, path = "/{id}/$config", tag = "Artifacts", params(("id" = String, Path)),
    responses((status = OK, body = ConfigOut), (status = NOT_FOUND, body = Problem)))]
async fn config(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<axum::Json<ConfigOut>> {
    fetch(&s, &id).await?;
    s.store()
        .config_for_artifact(&id)
        .await
        .map_err(store_problem)?
        .map(|c| axum::Json(c.into()))
        .ok_or_else(|| Problem::not_found(format!("No config linked to artifact {id}")))
}

/// The artifact's `data.metadata`.
#[utoipa::path(get, path = "/{id}/$metadata", tag = "Artifacts", params(("id" = String, Path)),
    responses((status = OK, body = Object), (status = NOT_FOUND, body = Problem)))]
async fn metadata(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<axum::Json<Value>> {
    let a = fetch(&s, &id).await?;
    Ok(axum::Json(
        a.data.get("metadata").cloned().unwrap_or(Value::Null),
    ))
}

/// Download the artifact's content (checkpoint bytes, or the prediction frame as JSON records).
#[utoipa::path(get, path = "/{id}/$download", tag = "Artifacts", params(("id" = String, Path)),
    responses((status = OK, body = Vec<u8>), (status = NOT_FOUND, body = Problem)))]
async fn download(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Response> {
    let a = fetch(&s, &id).await?;
    let content_type = str_field(&a.data, "content_type")
        .unwrap_or("application/octet-stream")
        .to_owned();
    if let Some(bytes) = s
        .store()
        .artifact_content(&id)
        .await
        .map_err(store_problem)?
    {
        let disposition = format!("attachment; filename=artifact_{id}.safetensors");
        return Ok((
            [
                (header::CONTENT_TYPE, content_type),
                (header::CONTENT_DISPOSITION, disposition),
            ],
            bytes,
        )
            .into_response());
    }
    let content = a
        .data
        .get("content")
        .cloned()
        .filter(|c| !c.is_null())
        .ok_or_else(|| Problem::not_found(format!("Artifact {id} has no content")))?;
    let records = records(&content).unwrap_or(content);
    let disposition = format!("attachment; filename=artifact_{id}.json");
    Ok((
        [
            (header::CONTENT_TYPE, "application/json".to_owned()),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        axum::Json(records),
    )
        .into_response())
}

/// `{columns, data}` as a list of records.
fn records(frame: &Value) -> Option<Value> {
    let columns = frame.get("columns")?.as_array()?;
    let rows = frame.get("data")?.as_array()?;
    Some(Value::Array(
        rows.iter()
            .filter_map(Value::as_array)
            .map(|row| {
                Value::Object(
                    columns
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .zip(row.iter().cloned())
                        .collect(),
                )
            })
            .collect(),
    ))
}

/// The router (mount at `/api/v1/artifacts`).
pub fn router(state: AppState) -> OpenApiRouter {
    OpenApiRouter::new()
        .routes(routes!(list, create))
        .routes(routes!(schema))
        .routes(routes!(stats))
        .routes(routes!(get, update, delete))
        .routes(routes!(expand))
        .routes(routes!(tree))
        .routes(routes!(config))
        .routes(routes!(metadata))
        .routes(routes!(download))
        .with_state(state)
}
