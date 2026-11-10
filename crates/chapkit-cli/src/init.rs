//! `chapkit-rs init`: a new Rust model package from the one template.

use std::path::{Component, Path, PathBuf};

const CHAPKIT_GIT: &str = "https://github.com/winterop-com/chapkit-rs";

/// Template files, written with `{{placeholders}}` filled in. The Dockerfile is chapkit-rs's shared
/// one, so a new package builds exactly like the published models.
const FILES: &[(&str, &str)] = &[
    ("Cargo.toml", include_str!("../templates/Cargo.toml.tmpl")),
    ("src/model.rs", include_str!("../templates/model.rs.tmpl")),
    ("src/lib.rs", include_str!("../templates/lib.rs.tmpl")),
    ("src/main.rs", include_str!("../templates/main.rs.tmpl")),
    ("tests/model.rs", include_str!("../templates/test.rs.tmpl")),
    ("README.md", include_str!("../templates/README.md.tmpl")),
    (".gitignore", include_str!("../templates/gitignore.tmpl")),
    ("compose.yml", include_str!("../templates/compose.yml.tmpl")),
    ("Dockerfile", include_str!("../../../docker/Dockerfile")),
    (
        "Dockerfile.dockerignore",
        include_str!("../../../docker/Dockerfile.dockerignore"),
    ),
];

/// The names a package is known by.
#[derive(Debug, PartialEq, Eq)]
pub struct Names {
    /// Package, binary and service id: `my-model`.
    pub name: String,
    /// Rust crate: `my_model`.
    pub crate_name: String,
    /// Model type: `MyModel`.
    pub type_name: String,
    /// Display name: `My model`.
    pub title: String,
}

impl Names {
    /// Check `name` is a valid service id (lowercase words joined by hyphens) and derive the rest.
    pub fn new(name: &str, title: Option<String>) -> Result<Self, String> {
        let words: Vec<&str> = name.split('-').collect();
        let valid = words.iter().all(|w| {
            !w.is_empty()
                && w.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        }) && name.starts_with(|c: char| c.is_ascii_lowercase());
        if !valid {
            return Err(format!(
                "'{name}' is not a valid name: use lowercase letters and digits in words joined by hyphens, starting with a letter (for example my-model)"
            ));
        }
        let capitalize = |w: &str| {
            let mut c = w.chars();
            c.next()
                .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                .unwrap_or_default()
        };
        Ok(Self {
            name: name.to_owned(),
            crate_name: name.replace('-', "_"),
            type_name: words.iter().map(|w| capitalize(w)).collect(),
            title: title.unwrap_or_else(|| capitalize(&words.join(" "))),
        })
    }
}

/// Where the chapkit dependency comes from, and how Docker then builds the package.
struct Wiring {
    /// chapkit-rs from git: the package builds on its own, so GitHub can publish it.
    from_git: bool,
    dependency: String,
    docker_context: String,
    dockerfile: String,
    docker_workspace: String,
}

fn wiring(dir: &Path, chapkit_path: Option<&Path>) -> Result<Wiring, String> {
    let Some(chapkit) = chapkit_path else {
        // From git, pinned to the release of this command, so a package builds the same library until
        // its author moves the tag on. The package alone is the build context.
        return Ok(Wiring {
            from_git: true,
            dependency: format!(
                "chapkit = {{ git = \"{CHAPKIT_GIT}\", tag = \"v{}\" }}",
                env!("CARGO_PKG_VERSION")
            ),
            docker_context: ".".into(),
            dockerfile: "Dockerfile".into(),
            docker_workspace: ".".into(),
        });
    };
    let chapkit = chapkit
        .canonicalize()
        .map_err(|e| format!("{}: {e}", chapkit.display()))?;
    if !chapkit.join("crates/chapkit-core").is_dir() {
        return Err(format!(
            "{} does not look like a chapkit-rs checkout",
            chapkit.display()
        ));
    }
    let package = absolute(dir)?;
    // A relative path when the two share a folder below the root, else the absolute path.
    let shared = package
        .components()
        .zip(chapkit.components())
        .take_while(|(a, b)| a == b)
        .count();
    let dependency_path = if shared > 1 {
        relative_path(&package, &chapkit)
    } else {
        chapkit.clone()
    };
    let dir_name = package
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_owned();
    // With chapkit-rs next to the package, the parent directory is the build context.
    Ok(Wiring {
        from_git: false,
        dependency: format!("chapkit = {{ path = \"{}\" }}", dependency_path.display()),
        docker_context: "..".into(),
        dockerfile: format!("{dir_name}/Dockerfile"),
        docker_workspace: dir_name,
    })
}

/// Create the package. Refuses to write into a non-empty directory.
pub fn run(
    name: &str,
    dir: Option<PathBuf>,
    chapkit_path: Option<&Path>,
    title: Option<String>,
) -> Result<(), String> {
    let names = Names::new(name, title)?;
    let dir = dir.unwrap_or_else(|| PathBuf::from(&names.name));
    if dir.exists()
        && std::fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .next()
            .is_some()
    {
        return Err(format!("{} exists and is not empty", dir.display()));
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let wiring = wiring(&dir, chapkit_path)?;
    for (path, template) in FILES {
        let target = dir.join(path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let text = render(template, &names, &wiring);
        std::fs::write(&target, text).map_err(|e| format!("{}: {e}", target.display()))?;
    }
    // GitHub can only build a package that gets chapkit-rs from git.
    if wiring.from_git {
        let target = dir.join(".github/workflows/publish.yml");
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let text = render(
            include_str!("../templates/publish.yml.tmpl"),
            &names,
            &wiring,
        );
        std::fs::write(&target, text).map_err(|e| format!("{}: {e}", target.display()))?;
    }
    println!("Created {} in {}", names.title, dir.display());
    println!();
    println!("  cd {}", dir.display());
    println!("  cargo test                  # the model's tests");
    println!("  cargo run --release -- serve  # serve on http://127.0.0.1:8000");
    println!("  cargo run --release -- test   # in another terminal: chap-core's flow against it");
    println!();
    println!("The calculation to replace with your model is in src/model.rs.");
    if !wiring.from_git {
        println!(
            "No publishing workflow: GitHub cannot build against a local chapkit-rs checkout."
        );
    }
    Ok(())
}

fn render(template: &str, n: &Names, w: &Wiring) -> String {
    template
        .replace("{{name}}", &n.name)
        .replace("{{crate_name}}", &n.crate_name)
        .replace("{{type_name}}", &n.type_name)
        .replace("{{title}}", &n.title)
        .replace("{{chapkit_dependency}}", &w.dependency)
        .replace("{{docker_context}}", &w.docker_context)
        .replace("{{dockerfile}}", &w.dockerfile)
        .replace("{{docker_workspace}}", &w.docker_workspace)
}

fn absolute(p: &Path) -> Result<PathBuf, String> {
    p.canonicalize()
        .map_err(|e| format!("{}: {e}", p.display()))
}

/// `to` relative to `from` (both absolute), for a Cargo path dependency.
fn relative_path(from: &Path, to: &Path) -> PathBuf {
    let (f, t): (Vec<Component>, Vec<Component>) =
        (from.components().collect(), to.components().collect());
    let common = f.iter().zip(&t).take_while(|(a, b)| a == b).count();
    let mut out = PathBuf::new();
    for _ in common..f.len() {
        out.push("..");
    }
    for c in t.iter().skip(common) {
        out.push(c.as_os_str());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_validated_and_derived() {
        let n = Names::new("dengue-lag-2", None).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(n.crate_name, "dengue_lag_2");
        assert_eq!(n.type_name, "DengueLag2");
        assert_eq!(n.title, "Dengue lag 2");
        for bad in ["", "My-model", "my_model", "-x", "x-", "2model", "a--b"] {
            assert!(Names::new(bad, None).is_err(), "{bad}");
        }
    }

    #[test]
    fn relative_paths() {
        assert_eq!(
            relative_path(Path::new("/a/rs/model"), Path::new("/a/rs/chapkit-rs")),
            PathBuf::from("../chapkit-rs")
        );
        assert_eq!(
            relative_path(Path::new("/a/x/y"), Path::new("/a/b")),
            PathBuf::from("../../b")
        );
    }

    #[test]
    fn writes_every_file_and_refuses_non_empty_dirs() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let dir = tmp.path().join("pkg");
        run("my-model", Some(dir.clone()), None, None).unwrap_or_else(|e| panic!("{e}"));
        for (path, _) in FILES {
            assert!(dir.join(path).is_file(), "{path}");
        }
        let cargo = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap_or_default();
        assert!(cargo.contains("name = \"my-model\""));
        assert!(cargo.contains(&format!(
            "chapkit = {{ git = \"{CHAPKIT_GIT}\", tag = \"v{}\" }}",
            env!("CARGO_PKG_VERSION")
        )));
        let lib = std::fs::read_to_string(dir.join("src/lib.rs")).unwrap_or_default();
        assert!(lib.contains("pub struct MyModel;"));
        assert!(!lib.contains("{{"), "unfilled placeholder");
        let publish =
            std::fs::read_to_string(dir.join(".github/workflows/publish.yml")).unwrap_or_default();
        assert!(publish.contains("BIN=my-model"));
        assert!(run("my-model", Some(dir), None, None).is_err());
    }
}
