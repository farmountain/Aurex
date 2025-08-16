//! Backend dispatch layer routing operations to device implementations.

pub mod dispatch;
pub mod vulkan_backend;
pub mod sycl_backend;

pub use dispatch::{Backend, Dispatcher, Workload};
pub use vulkan_backend::VulkanBackend;
pub use sycl_backend::SyclBackend;
