use cudarc::driver::{CudaDevice, CudaSlice};
use psb_gpu_bench::prf::chacha::ChaChaCudaRng;

fn main() {
    divan::main();
}

#[divan::bench(
    args = [1024, 1024 * 1024, 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024],
    sample_count = 10
)]
fn chacha_rng(bencher: divan::Bencher, n: usize) {
    bencher
        .with_inputs(|| {
            let device = CudaDevice::new(0).unwrap();
            let rng = ChaChaCudaRng::init(n, device.clone(), [0u32; 8]);
            let buffer: CudaSlice<u32> = device.alloc_zeros(n).unwrap();
            let stream = device.fork_default_stream().unwrap();
            (rng, buffer, device, stream, n)
        })
        .input_counter(move |_| divan::counter::ItemsCount::new(n))
        .bench_values(|(mut rng, mut buffer, device, stream, _n)| {
            rng.fill_rng_into(&mut buffer.slice_mut(..), &stream);
            device.wait_for(&stream).unwrap();
        });
}
