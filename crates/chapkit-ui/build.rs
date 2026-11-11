//! Rebuild when the UI build changes: files added to or removed from `dist/` are not otherwise
//! seen by cargo.

fn main() {
    println!("cargo:rerun-if-changed=dist");
}
