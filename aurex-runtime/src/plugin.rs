#![allow(improper_ctypes_definitions)]

use libloading::{Library, Symbol};
use std::collections::HashMap;

/// Trait implemented by backend plugins.
pub trait BackendPlugin: Send + Sync {
    /// Name used to register the plugin.
    fn name(&self) -> &'static str;
    /// Called once when the plugin is loaded.
    fn initialize(&self);
    /// Execute the plugin's main functionality.
    fn execute(&self);
}

// Signature of the plugin constructor function exported by dynamic libraries.
type PluginCreate = unsafe extern "C" fn() -> *mut dyn BackendPlugin;

/// Registry that loads backend plugins dynamically and stores them by name.
pub struct PluginRegistry {
    plugins: HashMap<String, Box<dyn BackendPlugin>>,
    // Hold libraries to ensure they remain loaded for the lifetime of the registry.
    libs: Vec<Library>,
}

impl PluginRegistry {
    /// Create an empty plugin registry.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            libs: Vec::new(),
        }
    }

    /// Load a plugin dynamic library from `path` and register it.
    ///
    /// # Safety
    ///
    /// Loading arbitrary dynamic libraries is inherently unsafe. The caller must
    /// ensure the library is trusted and follows the expected ABI.
    pub unsafe fn load(&mut self, path: &str) -> Result<(), libloading::Error> {
        let lib = Library::new(path)?;
        let constructor: Symbol<PluginCreate> = lib.get(b"create_plugin")?;
        let plugin = Box::<dyn BackendPlugin>::from_raw(constructor());
        plugin.initialize();
        let name = plugin.name().to_string();
        self.plugins.insert(name, plugin);
        self.libs.push(lib);
        Ok(())
    }

    /// Execute a previously loaded plugin by name.
    pub fn execute(&self, name: &str) {
        if let Some(plugin) = self.plugins.get(name) {
            plugin.execute();
        } else {
            eprintln!("Plugin '{}' not found", name);
        }
    }

    /// List the names of all loaded plugins.
    pub fn list(&self) -> Vec<&str> {
        self.plugins.keys().map(|k| k.as_str()).collect()
    }
}
