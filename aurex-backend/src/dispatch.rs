//! Backend dispatch stub.

#[derive(Debug)]
pub enum Backend {
    Cpu,
    #[allow(dead_code)]
    Rocm,
    #[allow(dead_code)]
    Sycl,
    #[allow(dead_code)]
    OpenCl,
}

impl Backend {
    pub fn execute(&self) {
        match self {
            Backend::Cpu => println!("Executing on CPU"),
            Backend::Rocm => println!("Executing on ROCm"),
            Backend::Sycl => println!("Executing on SYCL via oneAPI"),
            Backend::OpenCl => println!("Executing on OpenCL"),
        }
    }
}
