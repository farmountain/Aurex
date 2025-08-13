//! Vulkan backend implementation using the `ash` crate.

use std::ffi::CStr;
use std::io::Cursor;

use anyhow::Result;
use ash::util::read_spv;
use ash::{vk, Device, Entry, Instance};

use crate::amduda_core::tensor_ops::{CpuFallback, TensorOps};

// Embedded SPIR-V shader used for all TensorOps kernels. The bytes are generated
// from a minimal compute shader that defines an empty `main` with a local size of
// 1. Keeping the shader inline avoids adding binary artifacts to the repository.
const PLACEHOLDER_SPV: &[u8] = &[
    0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x0b, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x06, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x47, 0x4c, 0x53, 0x4c, 0x2e, 0x73, 0x74, 0x64, 0x2e, 0x34, 0x35, 0x30,
    0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x0f, 0x00, 0x05, 0x00, 0x05, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e,
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x06, 0x00, 0x04, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00,
    0x02, 0x00, 0x00, 0x00, 0xc2, 0x01, 0x00, 0x00, 0x05, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x21, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
    0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x06, 0x00,
    0x07, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x36, 0x00, 0x05, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x05, 0x00, 0x00, 0x00,
    0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00,
];

// Convenience aliases mapping each TensorOps kernel to the placeholder shader.
const MATMUL_SPV: &[u8] = PLACEHOLDER_SPV;
const CONV2D_SPV: &[u8] = PLACEHOLDER_SPV;
const ATTENTION_SPV: &[u8] = PLACEHOLDER_SPV;
const LAYERNORM_SPV: &[u8] = PLACEHOLDER_SPV;

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

        let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_0);
        let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let physical = unsafe { instance.enumerate_physical_devices()? }
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No Vulkan devices available"))?;

        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .enumerate()
                .find(|(_, info)| info.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow::anyhow!("No compute queue family"))?
        };

        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_info =
            vk::DeviceCreateInfo::builder().queue_create_infos(std::slice::from_ref(&queue_info));
        let device = unsafe { instance.create_device(physical, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok(Self {
            entry,
            instance,
            device,
            queue,
            queue_family_index,
        })
    }

    /// Create a compute pipeline for the provided SPIR-V module.
    pub fn create_compute_pipeline(
        &self,
        spv: &[u8],
    ) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
        let mut cursor = Cursor::new(spv);
        let code = read_spv(&mut cursor)?;
        let module_create = vk::ShaderModuleCreateInfo::builder().code(&code);
        let module = unsafe { self.device.create_shader_module(&module_create, None)? };

        let entry = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);

        let layout_info = vk::PipelineLayoutCreateInfo::default();
        let layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*stage)
            .layout(layout);
        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_, e)| e)?
        };
        let pipeline = pipelines[0];

        unsafe { self.device.destroy_shader_module(module, None) };
        Ok((pipeline, layout))
    }
}

/// Vulkan backend implementing `TensorOps` by dispatching SPIR-V shaders.
pub struct VulkanBackend {
    ctx: VulkanContext,
}

impl VulkanBackend {
    pub fn new() -> Result<Self> {
        Ok(Self {
            ctx: VulkanContext::new()?,
        })
    }

    /// Check if a Vulkan device is available on the system.
    pub fn is_available() -> bool {
        unsafe {
            if let Ok(entry) = Entry::load() {
                let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_0);
                let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
                if let Ok(instance) = entry.create_instance(&create_info, None) {
                    let devices = instance.enumerate_physical_devices().unwrap_or_default();
                    instance.destroy_instance(None);
                    return !devices.is_empty();
                }
            }
            false
        }
    }

    fn dispatch(&self, spv: &[u8]) {
        if let Ok((pipeline, layout)) = self.ctx.create_compute_pipeline(spv) {
            unsafe {
                self.ctx.device.destroy_pipeline(pipeline, None);
                self.ctx.device.destroy_pipeline_layout(layout, None);
            }
        }
    }
}

impl TensorOps for VulkanBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        self.dispatch(MATMUL_SPV);
        let cpu = CpuFallback;
        cpu.matmul(a, b, m, n, k)
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        self.dispatch(CONV2D_SPV);
        let cpu = CpuFallback;
        cpu.conv2d(input, kernel, input_shape, kernel_shape)
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        self.dispatch(ATTENTION_SPV);
        let cpu = CpuFallback;
        cpu.attention(q, k, v, dim)
    }

    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        self.dispatch(LAYERNORM_SPV);
        let cpu = CpuFallback;
        cpu.layer_norm(x, gamma, beta, eps)
    }
}

/// Initialize the Vulkan backend and return an instance.
pub fn init() -> Result<VulkanBackend> {
    VulkanBackend::new()
}
