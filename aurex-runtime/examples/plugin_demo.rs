use aurex_runtime::PluginRegistry;
use std::path::PathBuf;

fn main() {
    let mut registry = PluginRegistry::new();
    // Build path to the plugin shared library within the workspace target directory.
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("..");
    path.push("target");
    path.push("debug");
    path.push(format!(
        "{}fpga_npu.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    ));
    let path_str = path.to_string_lossy().into_owned();

    unsafe {
        registry.load(&path_str).expect("failed to load plugin");
    }
    // Execute the plugin once loaded.
    registry.execute("fpga_npu");
}
