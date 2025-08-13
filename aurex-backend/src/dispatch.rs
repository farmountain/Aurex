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
        println!("Executing on {:?}", self);
    }
}
