//! `chapkit-rs artifact`: list and download what a service stored, from the running service
//! (`--url`) or straight from its database file (`--database`), as Python chapkit's
//! `chapkit artifact` does.

use std::collections::BTreeMap;
use std::path::PathBuf;

use chapkit_store::Store;
use serde_json::Value;

/// `chapkit-rs artifact` subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// List artifacts as a tree
    List {
        #[command(flatten)]
        source: Source,
        /// Only this type, e.g. ml_prediction
        #[arg(long, short)]
        r#type: Option<String>,
    },
    /// Save an artifact's content
    ///
    /// A trained model is saved as its safetensors checkpoint, a forecast as JSON records.
    Download {
        /// Artifact id
        id: String,
        #[command(flatten)]
        source: Source,
        /// Output file [default: artifact_ID.safetensors or .json]
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Overwrite the output file
        #[arg(long)]
        force: bool,
    },
}

/// Where the artifacts are: a running service, or its database file.
#[derive(Debug, clap::Args)]
#[group(required = true, multiple = false)]
pub struct Source {
    /// Service URL
    #[arg(long, short)]
    url: Option<String>,
    /// SQLite database file
    #[arg(long, short)]
    database: Option<PathBuf>,
}

/// One artifact, as both sources give it.
struct Row {
    id: String,
    parent_id: Option<String>,
    data: Value,
    created_at: String,
}

pub async fn run(command: Command) -> Result<(), String> {
    match command {
        Command::List { source, r#type } => list(&source, r#type.as_deref()).await,
        Command::Download {
            id,
            source,
            output,
            force,
        } => download(&id, &source, output, force).await,
    }
}

async fn rows(source: &Source) -> Result<Vec<Row>, String> {
    if let Some(url) = &source.url {
        let url = format!("{}/api/v1/artifacts", url.trim_end_matches('/'));
        let list: Vec<Value> = get_json(&url)
            .await?
            .as_array()
            .cloned()
            .unwrap_or_default();
        return Ok(list
            .into_iter()
            .map(|a| Row {
                id: text(&a, "id"),
                parent_id: a
                    .get("parent_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                created_at: text(&a, "created_at"),
                data: a.get("data").cloned().unwrap_or(Value::Null),
            })
            .collect());
    }
    let store = open(source)?;
    let (list, _) = store
        .list_artifacts(None)
        .await
        .map_err(|e| e.to_string())?;
    Ok(list
        .into_iter()
        .map(|a| Row {
            id: a.id,
            parent_id: a.parent_id,
            data: a.data,
            created_at: a.created_at,
        })
        .collect())
}

fn open(source: &Source) -> Result<chapkit_store::SqliteStore, String> {
    let path = source.database.as_ref().ok_or("no --url or --database")?;
    if !path.is_file() {
        return Err(format!("{}: no such database file", path.display()));
    }
    chapkit_store::open_sqlite(Some(&path.to_string_lossy())).map_err(|e| e.to_string())
}

async fn get_json(url: &str) -> Result<Value, String> {
    let res = reqwest::get(url).await.map_err(|e| format!("{url}: {e}"))?;
    if !res.status().is_success() {
        return Err(format!("{url}: HTTP {}", res.status()));
    }
    res.json().await.map_err(|e| format!("{url}: {e}"))
}

fn text(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

async fn list(source: &Source, only: Option<&str>) -> Result<(), String> {
    let rows = rows(source).await?;
    let kind = |r: &Row| text(&r.data, "type");
    let shown: Vec<&Row> = rows
        .iter()
        .filter(|r| only.is_none_or(|t| kind(r) == t))
        .collect();
    if shown.is_empty() {
        println!("No artifacts");
        return Ok(());
    }
    // Children under their parent, indented; a filtered list is flat.
    let mut children: BTreeMap<Option<&str>, Vec<&Row>> = BTreeMap::new();
    for r in &shown {
        let parent = r
            .parent_id
            .as_deref()
            .filter(|p| only.is_none() && shown.iter().any(|s| s.id == *p));
        children.entry(parent).or_default().push(r);
    }
    println!(
        "{:<32} {:<15} {:>10}  {:<26}  CREATED",
        "ID", "TYPE", "SIZE", "CONFIG"
    );
    let mut stack: Vec<(&Row, usize)> = children
        .get(&None)
        .into_iter()
        .flatten()
        .rev()
        .map(|r| (*r, 0))
        .collect();
    while let Some((r, depth)) = stack.pop() {
        let config = r
            .data
            .get("metadata")
            .map(|m| text(m, "config_id"))
            .filter(|c| !c.is_empty());
        // Forecasts carry no size: measure their content.
        let size = r
            .data
            .get("content_size")
            .and_then(Value::as_u64)
            .or_else(|| {
                let content = r.data.get("content").filter(|c| !c.is_null())?;
                serde_json::to_vec(content).ok().map(|b| b.len() as u64)
            })
            .map_or("-".into(), format_size);
        println!(
            "{:<32} {:<15} {:>10}  {:<26}  {}",
            format!("{}{}", "  ".repeat(depth), r.id),
            kind(r),
            size,
            config.unwrap_or_else(|| "-".into()),
            r.created_at
        );
        for child in children
            .get(&Some(r.id.as_str()))
            .into_iter()
            .flatten()
            .rev()
        {
            stack.push((child, depth + 1));
        }
    }
    Ok(())
}

fn format_size(bytes: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} B")
    } else if b < 1024.0 * 1024.0 {
        format!("{:.1} KB", b / 1024.0)
    } else {
        format!("{:.1} MB", b / 1024.0 / 1024.0)
    }
}

async fn download(
    id: &str,
    source: &Source,
    output: Option<PathBuf>,
    force: bool,
) -> Result<(), String> {
    let (bytes, extension) = content(id, source).await?;
    let path = output.unwrap_or_else(|| PathBuf::from(format!("artifact_{id}.{extension}")));
    if path.exists() && !force {
        return Err(format!(
            "{} exists; use --force to overwrite",
            path.display()
        ));
    }
    std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    println!(
        "Saved {} ({}) to {}",
        id,
        format_size(bytes.len() as u64),
        path.display()
    );
    Ok(())
}

/// The artifact's content and the file extension for it: a checkpoint, or JSON records.
async fn content(id: &str, source: &Source) -> Result<(Vec<u8>, &'static str), String> {
    if let Some(url) = &source.url {
        let url = format!(
            "{}/api/v1/artifacts/{id}/$download",
            url.trim_end_matches('/')
        );
        let res = reqwest::get(&url)
            .await
            .map_err(|e| format!("{url}: {e}"))?;
        if !res.status().is_success() {
            return Err(format!("{url}: HTTP {}", res.status()));
        }
        let json = res
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|t| t.contains("json"));
        let bytes = res.bytes().await.map_err(|e| e.to_string())?.to_vec();
        return Ok((bytes, if json { "json" } else { "safetensors" }));
    }
    let store = open(source)?;
    if let Some(blob) = store
        .artifact_content(id)
        .await
        .map_err(|e| e.to_string())?
    {
        return Ok((blob, "safetensors"));
    }
    let row = store
        .get_artifact(id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("artifact {id} not found"))?;
    let content = row
        .data
        .get("content")
        .filter(|c| !c.is_null())
        .ok_or_else(|| format!("artifact {id} has no content"))?;
    let bytes = serde_json::to_vec_pretty(&records(content)).map_err(|e| e.to_string())?;
    Ok((bytes, "json"))
}

/// A `{columns, data}` table as JSON records, `[{column: value}, ...]`, the form the service's
/// `$download` gives; anything else unchanged.
fn records(content: &Value) -> Value {
    let (Some(columns), Some(rows)) = (
        content.get("columns").and_then(Value::as_array),
        content.get("data").and_then(Value::as_array),
    ) else {
        return content.clone();
    };
    Value::Array(
        rows.iter()
            .filter_map(Value::as_array)
            .map(|row| {
                let record: serde_json::Map<String, Value> = columns
                    .iter()
                    .filter_map(Value::as_str)
                    .zip(row)
                    .map(|(c, v)| (c.to_owned(), v.clone()))
                    .collect();
                Value::Object(record)
            })
            .collect(),
    )
}
