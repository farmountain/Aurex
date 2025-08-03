//! Backend dispatch stub.

#[derive(Debug)]
pub enum Backend {
    Cpu,
    #[allow(dead_code)]
    Rocm,
    #[allow(dead_code)]
    Sycl,
}

impl Backend {
    pub fn execute(&self) {
        println!("Executing on {:?}", self);
    }
}
