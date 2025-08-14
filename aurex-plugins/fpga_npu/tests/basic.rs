use aurex_runtime::BackendPlugin;
use fpga_npu::create_plugin;

#[test]
fn plugin_creates_and_reports_name() {
    let plugin: Box<dyn BackendPlugin> = unsafe { Box::from_raw(create_plugin()) };
    assert_eq!(plugin.name(), "fpga_npu");
}
