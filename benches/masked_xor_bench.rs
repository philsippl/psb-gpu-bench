use psb_gpu_bench::masked_xor::MaskedXor;
use rand::random;

const ARGS: &[u64] = &[1024, 1024 * 1024, 16 * 1024 * 1024];

fn main() {
    divan::main();
}

fn setup_and_bench(bencher: divan::Bencher, n: u64, limbs: usize) {
    let host_a: Vec<u64> = (0..(n * limbs as u64)).map(|_| random()).collect();
    let host_b: Vec<u64> = (0..(n * limbs as u64)).map(|_| random()).collect();

    let instance = MaskedXor::new(limbs);
    let device = instance.device();
    let a = device.htod_copy(host_a).unwrap();
    let b = device.htod_copy(host_b).unwrap();

    bencher
        .with_inputs(move || (a.clone(), b.clone()))
        .input_counter(move |_| divan::counter::ItemsCount::new(n * 2))
        .bench_values(|(a, b)| instance.run_with_device_ptrs(&a, &b));
}

#[divan::bench(args = ARGS, sample_count = 10)]
fn masked_xor_64(bencher: divan::Bencher, n: u64) {
    setup_and_bench(bencher, n, 1);
}

#[divan::bench(args = ARGS, sample_count = 10)]
fn masked_xor_128(bencher: divan::Bencher, n: u64) {
    setup_and_bench(bencher, n, 2);
}

#[divan::bench(args = ARGS, sample_count = 10)]
fn masked_xor_256(bencher: divan::Bencher, n: u64) {
    setup_and_bench(bencher, n, 4);
}
