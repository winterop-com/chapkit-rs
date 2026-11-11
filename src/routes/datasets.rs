//! `/api/v1/datasets`: datasets stored in the service, so backtests and forecasts can name one
//! instead of sending it. chapkit-rs extension; not part of Python chapkit.

use std::collections::{BTreeSet, HashSet};

use axum::extract::{FromRequest, Path, RawQuery, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use chapkit_core::{Column, DataFrame, Period, cols};
use chapkit_http::time::entity_now;
use chapkit_http::{ApiResult, Json, ListOrPage, Problem, Query, parse_ulid};
use chapkit_store::{DatasetRow, Page};
use serde_json::json;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use crate::state::{AppState, store_problem};
use crate::wire::{DatasetIn, DatasetMeta, DatasetOut, DatasetUpload};

const PREFIX: &str = "/api/v1/datasets";

fn not_found(id: &str) -> Problem {
    Problem::not_found(format!("Dataset with id {id} not found")).instance(format!("{PREFIX}/{id}"))
}

fn invalid(field: &str, msg: impl Into<String>) -> Problem {
    Problem::validation(vec![
        json!({"loc": ["body", field], "msg": msg.into(), "type": "value_error"}),
    ])
}

fn out(row: DatasetRow) -> ApiResult<DatasetOut> {
    let meta: DatasetMeta = serde_json::from_value(row.meta).map_err(|e| {
        tracing::error!(error = %e, id = %row.id, "dataset.meta");
        Problem::internal()
    })?;
    Ok(DatasetOut {
        id: row.id,
        name: row.name,
        created_at: row.created_at,
        updated_at: row.updated_at,
        meta,
    })
}

/// A location as text (numbers as written), `None` when missing.
fn location_at(col: &Column, row: usize) -> Option<String> {
    col.str_at(row).map(str::to_owned).or_else(|| {
        let v = col.json_at(row);
        (!v.is_null()).then(|| v.to_string())
    })
}

/// Check a dataset and describe it: `time_period` (every chap-core form) and `location` on every
/// row, one kind of period, no location twice in a period, and a numeric `target`.
pub(crate) fn describe(data: &DataFrame, target: &str) -> Result<DatasetMeta, String> {
    if data.is_empty() || data.n_rows() == 0 {
        return Err("the dataset has no rows".into());
    }
    let tp = data
        .column(cols::TIME_PERIOD)
        .ok_or("the dataset has no 'time_period' column")?;
    let loc = data
        .column(cols::LOCATION)
        .ok_or("the dataset has no 'location' column")?;
    match data.column(target) {
        None => return Err(format!("the dataset has no target column '{target}'")),
        Some(Column::Float(_) | Column::Int(_)) => {}
        Some(_) => return Err(format!("the target column '{target}' is not numeric")),
    }
    let mut periods = BTreeSet::new();
    let mut seen = HashSet::with_capacity(data.n_rows());
    let mut locations = BTreeSet::new();
    for row in 0..data.n_rows() {
        let label = tp.str_at(row).unwrap_or_default();
        let period = Period::parse(label).ok_or_else(|| {
            format!(
                "time_period {label:?} in row {} is not a month or week",
                row + 1
            )
        })?;
        let location =
            location_at(loc, row).ok_or_else(|| format!("row {} has no location", row + 1))?;
        if !seen.insert((location.clone(), period)) {
            return Err(format!(
                "location {location} has more than one row for {period}"
            ));
        }
        locations.insert(location);
        periods.insert(period);
    }
    let (Some(first), Some(last)) = (periods.first().copied(), periods.last().copied()) else {
        return Err("the dataset has no rows".into());
    };
    let weekly = matches!(first, Period::Week { .. });
    if weekly != matches!(last, Period::Week { .. }) {
        return Err("the dataset mixes monthly and weekly periods".into());
    }
    let covariates = data
        .iter()
        .filter(|(name, col)| {
            ![cols::TIME_PERIOD, cols::LOCATION, target].contains(name)
                && matches!(col, Column::Float(_) | Column::Int(_))
        })
        .map(|(name, _)| name.to_owned())
        .collect();
    Ok(DatasetMeta {
        target: target.to_owned(),
        rows: data.n_rows(),
        locations: locations.len(),
        location_ids: locations.into_iter().collect(),
        period_type: if weekly { "week" } else { "month" }.into(),
        first_period: first.to_string(),
        last_period: last.to_string(),
        periods: periods.len(),
        columns: data.names().to_vec(),
        covariates,
    })
}

/// A stored dataset's frame as given: parsed once (on a blocking thread), then shared from the
/// service's bounded cache ([`crate::state::FrameCache`]); stored datasets never change.
async fn stored_frame(
    s: &AppState,
    id: &str,
) -> ApiResult<(std::sync::Arc<DataFrame>, DatasetMeta)> {
    let id = parse_ulid(id)?.to_string();
    let row = s
        .store()
        .get_dataset(&id)
        .await
        .map_err(store_problem)?
        .ok_or_else(|| not_found(&id))?;
    let meta = out(row)?.meta;
    if let Some(frame) = s.frames().get(&id) {
        return Ok((frame, meta));
    }
    let bytes = s
        .store()
        .dataset_data(&id)
        .await
        .map_err(store_problem)?
        .ok_or_else(|| not_found(&id))?;
    let parsed = tokio::task::spawn_blocking(move || serde_json::from_slice::<DataFrame>(&bytes))
        .await
        .map_err(|_| Problem::internal())?;
    let frame = std::sync::Arc::new(parsed.map_err(|e| {
        tracing::error!(error = %e, %id, "dataset.data");
        Problem::internal()
    })?);
    s.frames().put(id, frame.clone());
    Ok((frame, meta))
}

/// The dataset of a `$backtest` or `$forecast`: `data`, or the stored `dataset_id` with its target
/// as `disease_cases`, the column chap-core's evaluation scores.
pub(crate) async fn request_data(
    s: &AppState,
    dataset_id: Option<&str>,
    data: Option<DataFrame>,
) -> ApiResult<(DataFrame, Option<String>)> {
    match (dataset_id, data) {
        (Some(_), Some(_)) => Err(Problem::bad_request("give dataset_id or data, not both")),
        (None, None) => Err(Problem::bad_request("data or dataset_id is required")),
        (None, Some(data)) => Ok((data, None)),
        (Some(id), None) => {
            let (frame, meta) = stored_frame(s, id).await?;
            let frame = with_target_as_cases(std::sync::Arc::unwrap_or_clone(frame), &meta.target)
                .map_err(|e| Problem::bad_request(e.to_string()))?;
            Ok((frame, Some(parse_ulid(id)?.to_string())))
        }
    }
}

/// `frame` with its `target` column named `disease_cases` (an existing `disease_cases` dropped).
fn with_target_as_cases(
    frame: DataFrame,
    target: &str,
) -> Result<DataFrame, chapkit_core::FrameError> {
    if target == cols::DISEASE_CASES {
        return Ok(frame);
    }
    DataFrame::from_columns(frame.iter().filter_map(|(name, col)| match name {
        n if n == cols::DISEASE_CASES => None,
        n if n == target => Some((cols::DISEASE_CASES.to_owned(), col.clone())),
        n => Some((n.to_owned(), col.clone())),
    }))
}

/// Parse an uploaded file: chapkit DataFrame JSON for a `.json` name or a body starting with `{`,
/// else CSV.
fn parse_file(bytes: &[u8], filename: Option<&str>) -> Result<DataFrame, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "the file is not UTF-8 text")?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let json = filename.is_some_and(|f| f.to_ascii_lowercase().ends_with(".json"))
        || text.trim_start().starts_with('{');
    if json {
        serde_json::from_str(text)
            .map_err(|e| format!("the file is not chapkit DataFrame JSON: {e}"))
    } else {
        DataFrame::from_csv(text).map_err(|e| format!("the file is not a readable CSV: {e}"))
    }
}

/// The fields of a multipart upload.
async fn upload(req: Request) -> ApiResult<(String, DataFrame, Option<String>)> {
    let too_large = |e: &axum::extract::multipart::MultipartError| {
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            Problem::payload_too_large(
                "Request body is larger than the service accepts (CHAPKIT_MAX_BODY_MB)",
            )
        } else {
            Problem::bad_request(e.body_text())
        }
    };
    let mut form = axum::extract::Multipart::from_request(req, &())
        .await
        .map_err(|e| Problem::bad_request(e.body_text()))?;
    let (mut file, mut name, mut target) = (None, None, None);
    while let Some(field) = form.next_field().await.map_err(|e| too_large(&e))? {
        match field.name().unwrap_or_default() {
            "file" => {
                let filename = field.file_name().map(str::to_owned);
                let bytes = field.bytes().await.map_err(|e| too_large(&e))?;
                file = Some((bytes, filename));
            }
            "name" => name = Some(field.text().await.map_err(|e| too_large(&e))?),
            "target" => target = Some(field.text().await.map_err(|e| too_large(&e))?),
            _ => {}
        }
    }
    let (bytes, filename) = file.ok_or_else(|| invalid("file", "Field required"))?;
    let frame = parse_file(&bytes, filename.as_deref()).map_err(Problem::bad_request)?;
    let name = name.filter(|n| !n.trim().is_empty()).or_else(|| {
        filename.map(|f| {
            std::path::Path::new(&f)
                .file_stem()
                .map_or(f.clone(), |s| s.to_string_lossy().into_owned())
        })
    });
    let name = name.ok_or_else(|| invalid("name", "Field required"))?;
    Ok((name, frame, target.filter(|t| !t.trim().is_empty())))
}

/// List datasets (paged when both `page` and `size` are given), only those whose name holds `q`
/// when given.
#[utoipa::path(get, path = "", tag = "Datasets", params(super::NamedListParams),
    responses((status = OK, body = ListOrPage<DatasetOut>), (status = UNPROCESSABLE_ENTITY, body = Problem)))]
async fn list(
    State(s): State<AppState>,
    Query(p): Query<super::NamedListParams>,
) -> ApiResult<ListOrPage<DatasetOut>> {
    let paging = p.paging()?;
    let page = paging.map(|(page, size)| Page {
        offset: chapkit_http::extract::page_offset(page, size),
        limit: size,
    });
    let (rows, total) = s
        .store()
        .list_datasets(p.name(), page)
        .await
        .map_err(store_problem)?;
    let items = rows.into_iter().map(out).collect::<ApiResult<Vec<_>>>()?;
    Ok(ListOrPage::new(items, total, paging))
}

/// Store a dataset: JSON `{name, data, target?}`, or a CSV or JSON `file` (with `name` and
/// `target`) as `multipart/form-data`. It is checked (periods in any form chap-core reads, a
/// location on every row, a numeric target) and stored as given.
#[utoipa::path(post, path = "", tag = "Datasets",
    request_body(content((DatasetIn = "application/json"), (DatasetUpload = "multipart/form-data"))),
    responses((status = CREATED, body = DatasetOut), (status = BAD_REQUEST, body = Problem),
        (status = UNPROCESSABLE_ENTITY, body = Problem)))]
async fn create(State(s): State<AppState>, req: Request) -> ApiResult<Response> {
    let multipart = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("multipart/form-data"));
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost")
        .to_owned();
    let (name, frame, target) = if multipart {
        upload(req).await?
    } else {
        let Json(body) = Json::<DatasetIn>::from_request(req, &()).await?;
        (body.name, body.data, body.target)
    };
    if name.trim().is_empty() {
        return Err(invalid("name", "The name must not be empty"));
    }
    let target = target.unwrap_or_else(|| cols::DISEASE_CASES.to_owned());
    let meta = describe(&frame, &target).map_err(Problem::bad_request)?;
    let data = serde_json::to_vec(&frame).map_err(|_| Problem::internal())?;
    let now = entity_now();
    let id = chapkit_http::ids::new_id().to_string();
    let meta_json = serde_json::to_value(&meta).map_err(|_| Problem::internal())?;
    let row = DatasetRow {
        id: id.clone(),
        name,
        meta: meta_json,
        created_at: now.clone(),
        updated_at: now,
    };
    s.store()
        .insert_dataset(row.clone(), data)
        .await
        .map_err(store_problem)?;
    let mut res = (StatusCode::CREATED, axum::Json(out(row)?)).into_response();
    if let Ok(v) = format!("http://{host}{PREFIX}/{id}").parse() {
        res.headers_mut().insert(header::LOCATION, v);
    }
    Ok(res)
}

/// Get a dataset's description.
#[utoipa::path(get, path = "/{id}", tag = "Datasets", params(("id" = String, Path)),
    responses((status = OK, body = DatasetOut), (status = NOT_FOUND, body = Problem), (status = BAD_REQUEST, body = Problem)))]
async fn get(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<axum::Json<DatasetOut>> {
    parse_ulid(&id)?;
    let row = s
        .store()
        .get_dataset(&id)
        .await
        .map_err(store_problem)?
        .ok_or_else(|| not_found(&id))?;
    out(row).map(axum::Json)
}

/// Delete a dataset.
#[utoipa::path(delete, path = "/{id}", tag = "Datasets", params(("id" = String, Path)),
    responses((status = NO_CONTENT), (status = NOT_FOUND, body = Problem)))]
async fn delete(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<StatusCode> {
    parse_ulid(&id)?;
    if s.store().delete_dataset(&id).await.map_err(store_problem)? {
        s.frames().remove(&id);
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(not_found(&id))
    }
}

/// Decode one `application/x-www-form-urlencoded` component (`+` and `%XX`).
fn decode_component(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while let Some(&b) = bytes.get(i) {
        let hex = |j: usize| {
            bytes
                .get(j)
                .and_then(|c| char::from(*c).to_digit(16))
                .and_then(|d| u8::try_from(d).ok())
        };
        match b {
            b'+' => out.push(b' '),
            b'%' => {
                if let (Some(h), Some(l)) = (hex(i + 1), hex(i + 2)) {
                    out.push(h * 16 + l);
                    i += 2;
                } else {
                    out.push(b);
                }
            }
            _ => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The values of a repeated query parameter.
fn query_values(query: Option<&str>, key: &str) -> Vec<String> {
    query
        .unwrap_or_default()
        .split('&')
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            (decode_component(k) == key).then(|| decode_component(v))
        })
        .collect()
}

/// What of a stored dataset `$data` returns.
#[derive(Debug, Default)]
struct Projection {
    locations: HashSet<String>,
    columns: Vec<String>,
    before: Option<Period>,
    last: Option<usize>,
    canonical: bool,
}

impl Projection {
    fn parse(query: Option<&str>) -> Result<Self, Problem> {
        let one = |key: &str| query_values(query, key).pop();
        let before = one("before")
            .map(|raw| {
                Period::parse(&raw).ok_or_else(|| {
                    Problem::bad_request(format!("before {raw:?} is not a month or week"))
                })
            })
            .transpose()?;
        let last = one("last")
            .map(|raw| {
                raw.trim()
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| {
                        Problem::bad_request(format!("last {raw:?} is not a positive number"))
                    })
            })
            .transpose()?;
        let canonical = match one("canonical").as_deref() {
            None | Some("false" | "0") => false,
            Some("true" | "1" | "") => true,
            Some(raw) => {
                return Err(Problem::bad_request(format!(
                    "canonical {raw:?} is not true or false"
                )));
            }
        };
        Ok(Self {
            locations: query_values(query, "location").into_iter().collect(),
            columns: query_values(query, "column"),
            before,
            last,
            canonical,
        })
    }

    /// Whether rows need their periods read.
    fn by_period(&self) -> bool {
        self.before.is_some() || self.last.is_some() || self.canonical
    }

    /// The projected frame: the rows of the locations asked for, before `before`, in the `last`
    /// periods; only `time_period`, `location` and the `columns` asked for; with `canonical`, the
    /// periods written as chapkit writes them and the rows in period order.
    fn apply<'a>(&self, frame: &'a DataFrame) -> Result<std::borrow::Cow<'a, DataFrame>, Problem> {
        let bad = |e: chapkit_core::FrameError| Problem::bad_request(e.to_string());
        let mut rows: Vec<usize> = (0..frame.n_rows()).collect();
        if !self.locations.is_empty() {
            let loc = frame.require(cols::LOCATION).map_err(bad)?;
            rows.retain(|&r| location_at(loc, r).is_some_and(|l| self.locations.contains(&l)));
        }
        let mut periods = Vec::new();
        if self.by_period() {
            let tp = frame.require(cols::TIME_PERIOD).map_err(bad)?;
            // Stored datasets were checked on upload: every period parses.
            let mut keyed: Vec<(Period, usize)> = rows
                .iter()
                .filter_map(|&r| tp.str_at(r).and_then(Period::parse).map(|p| (p, r)))
                .filter(|(p, _)| self.before.is_none_or(|b| *p < b))
                .collect();
            if let Some(n) = self.last {
                let distinct: BTreeSet<Period> = keyed.iter().map(|(p, _)| *p).collect();
                if let Some(&from) = distinct.iter().rev().nth(n.saturating_sub(1)) {
                    keyed.retain(|(p, _)| *p >= from);
                }
            }
            if self.canonical {
                keyed.sort_by_key(|(p, _)| *p);
            }
            rows = keyed.iter().map(|(_, r)| *r).collect();
            periods = keyed.into_iter().map(|(p, _)| p.to_string()).collect();
        }
        let mut out = if rows.len() == frame.n_rows() && !self.canonical {
            std::borrow::Cow::Borrowed(frame)
        } else {
            std::borrow::Cow::Owned(frame.take(&rows).map_err(bad)?)
        };
        if self.canonical {
            out.to_mut()
                .set_column(cols::TIME_PERIOD, Column::from_strs(periods))
                .map_err(bad)?;
        }
        if !self.columns.is_empty() {
            let mut keep: Vec<&str> = vec![cols::TIME_PERIOD, cols::LOCATION];
            for c in &self.columns {
                if !out.has(c) {
                    return Err(Problem::bad_request(format!(
                        "the dataset has no column '{c}'"
                    )));
                }
                if !keep.contains(&c.as_str()) {
                    keep.push(c);
                }
            }
            out = std::borrow::Cow::Owned(out.select(&keep).map_err(bad)?);
        }
        Ok(out)
    }
}

/// The dataset as stored, chapkit DataFrame JSON, or a projection of it: `location`
/// (repeatable) for some locations' rows, `column` (repeatable) for only those columns besides
/// `time_period` and `location`, `before` for the periods before one, `last` for the last periods,
/// and `canonical=true` for the periods as chapkit writes them (`2024-01`, `2024W03`, `2024SunW03`),
/// the rows in period order.
#[utoipa::path(get, path = "/{id}/$data", tag = "Datasets",
    params(("id" = String, Path),
        ("location" = Option<Vec<String>>, Query, explode, description = "Only these locations (repeatable)"),
        ("column" = Option<Vec<String>>, Query, explode, description = "Only these columns besides time_period and location (repeatable)"),
        ("before" = Option<String>, Query, description = "Only the periods before this one, in any form a dataset takes"),
        ("last" = Option<usize>, Query, description = "Only the last this many periods (after `before`)"),
        ("canonical" = Option<bool>, Query, description = "Periods as chapkit writes them, the rows in period order")),
    responses((status = OK, body = Object), (status = NOT_FOUND, body = Problem), (status = BAD_REQUEST, body = Problem)))]
async fn data(
    State(s): State<AppState>,
    Path(id): Path<String>,
    RawQuery(query): RawQuery,
) -> ApiResult<Response> {
    let projection = Projection::parse(query.as_deref())?;
    let (frame, _) = stored_frame(&s, &id).await?;
    Ok(axum::Json(projection.apply(&frame)?.as_ref()).into_response())
}

/// The router (mount at `/api/v1/datasets`).
pub fn router(state: AppState) -> OpenApiRouter {
    OpenApiRouter::new()
        .routes(routes!(list, create))
        .routes(routes!(get, delete))
        .routes(routes!(data))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_values_decode_and_repeat() {
        assert_eq!(
            query_values(Some("location=A+B&x=1&location=%C3%B8st%2F1"), "location"),
            ["A B", "øst/1"]
        );
        assert_eq!(query_values(None, "location"), Vec::<String>::new());
        assert_eq!(decode_component("100%"), "100%");
    }
}
