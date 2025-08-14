use aurex_backend::{Backend, Dispatcher, Workload};

#[test]
fn matmul_across_backends() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    for backend in [Backend::Cpu, Backend::Rocm, Backend::Sycl, Backend::OpenCl] {
        let dispatcher = Dispatcher::new(Some(backend), Workload::Light);
        assert_eq!(dispatcher.matmul(&a, &b, 2, 2, 2), expected);
    }
}
