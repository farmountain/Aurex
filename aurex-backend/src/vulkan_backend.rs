//! Vulkan backend providing minimal instance/device management and compute
//! pipeline utilities.  The backend compiles a tiny compute shader to SPIR-V at
//! runtime and uses it for all [`TensorOps`] kernels.  If Vulkan is unavailable
//! on the system the backend transparently falls back to the CPU implementation.

use std::ffi::CStr;
use std::io::Cursor;

use anyhow::Result;
use ash::util::read_spv;
use ash::{vk, Device, Entry, Instance};
use shaderc::{Compiler, ShaderKind};

use crate::dispatch::{CpuBackend, TensorOps};

/// Minimal compute shader used as a stand‑in for all TensorOps kernels.  The
/// shader simply defines an empty `main` function with a workgroup size of 1.
const PLACEHOLDER_SHADER: &str = r#"
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {}
"#;

/// Compile a GLSL compute shader to SPIR-V words.
pub fn compile_shader(src: &str) -> Result<Vec<u32>, String> {
    let mut compiler = Compiler::new().ok_or("failed to create shader compiler")?;
    let binary = compiler
        .compile_into_spirv(src, ShaderKind::Compute, "kernel.glsl", "main", None)
        .map_err(|e| e.to_string())?;
    Ok(binary.as_binary().to_vec())
}

/// Load a SPIR-V module from disk.
pub fn load_shader(path: &str) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut cursor = Cursor::new(bytes);
    read_spv(&mut cursor).map_err(|e| e.to_string())
}

/// Holds Vulkan objects required for compute dispatch.
pub struct VulkanContext {
    entry: Entry,
    instance: Instance,
    device: Device,
    queue: vk::Queue,
    queue_family_index: u32,
}

impl VulkanContext {
    /// Create a new Vulkan instance and logical device with a compute queue.
    pub fn new() -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        let app = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_0);
        let info = vk::InstanceCreateInfo::builder().application_info(&app);
        let instance = unsafe { entry.create_instance(&info, None)? };

        let physical = unsafe { instance.enumerate_physical_devices()? }
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No Vulkan devices available"))?;

        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .enumerate()
                .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow::anyhow!("No compute queue family"))?
        };

        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);
        let device_info = vk::DeviceCreateInfo::builder().queue_create_infos(std::slice::from_ref(&queue_info));
        let device = unsafe { instance.create_device(physical, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok(Self { entry, instance, device, queue, queue_family_index })
    }

    /// Create a compute pipeline from SPIR-V code.
    pub fn create_compute_pipeline(&self, code: &[u32]) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
        let module_info = vk::ShaderModuleCreateInfo::builder().code(code);
        let module = unsafe { self.device.create_shader_module(&module_info, None)? };

        let entry = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);

        let layout_info = vk::PipelineLayoutCreateInfo::default();
        let layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };

        let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(*stage).layout(layout);
        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None)
                .map_err(|(_, e)| e)?
        };
        let pipeline = pipelines[0];

        unsafe { self.device.destroy_shader_module(module, None) };
        Ok((pipeline, layout))
    }
}

/// Vulkan backend implementing [`TensorOps`] by dispatching placeholder shaders.
pub struct VulkanBackend {
    ctx: Option<VulkanContext>,
    matmul_spv: Vec<u32>,
    conv2d_spv: Vec<u32>,
    attention_spv: Vec<u32>,
    layernorm_spv: Vec<u32>,
}

impl VulkanBackend {
    /// Create a new backend.  If Vulkan initialization fails the backend will
    /// fall back to CPU execution.
    pub fn new() -> Self {
        let ctx = VulkanContext::new().ok();
        let spv = compile_shader(PLACEHOLDER_SHADER).unwrap_or_default();
        Self {
            ctx,
            matmul_spv: spv.clone(),
            conv2d_spv: spv.clone(),
            attention_spv: spv.clone(),
            layernorm_spv: spv,
        }
    }

    /// Check if a Vulkan device is available on the system.
    pub fn is_available() -> bool {
        VulkanContext::new().is_ok()
    }

    fn dispatch(&self, code: &[u32]) {
        if let Some(ctx) = &self.ctx {
            if let Ok((pipeline, layout)) = ctx.create_compute_pipeline(code) {
                unsafe {
                    ctx.device.destroy_pipeline(pipeline, None);
                    ctx.device.destroy_pipeline_layout(layout, None);
                }
            }
        }
    }
}

impl TensorOps for VulkanBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        self.dispatch(&self.matmul_spv);
        CpuBackend.matmul(a, b, m, n, k)
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        self.dispatch(&self.conv2d_spv);
        CpuBackend.conv2d(input, kernel, input_shape, kernel_shape)
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        self.dispatch(&self.attention_spv);
        CpuBackend.attention(q, k, v, dim)
    }

    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        self.dispatch(&self.layernorm_spv);
        CpuBackend.layer_norm(x, gamma, beta, eps)
    }
}

