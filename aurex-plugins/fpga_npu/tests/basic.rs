use aurex_runtime::BackendPlugin;
use fpga_npu::{create_plugin, driver};
use std::sync::atomic::Ordering;

#[test]
fn plugin_initializes_and_executes_driver() {
    let plugin: Box<dyn BackendPlugin> = unsafe { Box::from_raw(create_plugin()) };
    assert_eq!(plugin.name(), "fpga_npu");
    plugin.initialize();
    assert!(driver::INITIALIZED.load(Ordering::SeqCst));
    plugin.execute();
    assert!(driver::EXECUTED.load(Ordering::SeqCst));
}
