use cudarc::{
    driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};

fn ref_masked_xor(a: &[u64], b: &[u64], c: &mut [u64]) {
    for (a_, b_) in a.chunks(4).zip(b.chunks(4)) {
        c[0] ^= a_[0] & b_[0];
        c[1] ^= a_[1] & b_[1];
        c[2] ^= a_[2] & b_[2];
        c[3] ^= a_[3] & b_[3];
    }
}

fn main() {
    let NUM_ELEMENTS: usize = 1024;
    let BUFFER_SIZE: usize = NUM_ELEMENTS * 4;

    let device = CudaDevice::new(0).unwrap();
    let ptx = compile_ptx(include_str!("kernel.cu")).unwrap();
    device
        .load_ptx(
            ptx.clone(),
            "",
            &[
                "masked_xor_256",
                "masked_xor_256_x2",
                "masked_xor_256_x4",
                "masked_xor_256_x8",
            ],
        )
        .unwrap();
    let func = device.get_func("", "masked_xor_256").unwrap();
    let cfg = LaunchConfig {
        grid_dim: (NUM_ELEMENTS.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 32,
    };

    let mut a: CudaSlice<u64> = device.alloc_zeros(BUFFER_SIZE).unwrap();
    let mut b: CudaSlice<u64> = device.alloc_zeros(BUFFER_SIZE).unwrap();
    let mut c: CudaSlice<u64> = device.alloc_zeros(4).unwrap();

    let host_a: Vec<u64> = (0..BUFFER_SIZE).map(|_| rand::random::<u64>()).collect();
    let host_b: Vec<u64> = (0..BUFFER_SIZE).map(|_| rand::random::<u64>()).collect();
    device.htod_copy_into(host_a.to_vec(), &mut a).unwrap();
    device.htod_copy_into(host_b.to_vec(), &mut b).unwrap();

    unsafe {
        func.clone()
            .launch(cfg, (&mut a, &mut b, &mut c, NUM_ELEMENTS as u32))
    }
    .unwrap();

    let result_host = device.dtoh_sync_copy(&c).unwrap();
    println!("Result: {:?}", result_host);

    // reference
    let mut ref_c = [0u64; 4];
    ref_masked_xor(&host_a, &host_b, &mut ref_c);
    println!("Ref result: {:?}", ref_c);

    assert_eq!(result_host, ref_c);

    // bench
    let NUM_ELEMENTS: usize = 1024 * 1024 * 10;
    let BUFFER_SIZE: usize = NUM_ELEMENTS * 4;

    let mut a: CudaSlice<u64> = device.alloc_zeros(BUFFER_SIZE).unwrap();
    let mut b: CudaSlice<u64> = device.alloc_zeros(BUFFER_SIZE).unwrap();
    let mut c: CudaSlice<u64> = device.alloc_zeros(4).unwrap();
    let host_a: Vec<u64> = (0..BUFFER_SIZE).map(|_| rand::random::<u64>()).collect();
    let host_b: Vec<u64> = (0..BUFFER_SIZE).map(|_| rand::random::<u64>()).collect();
    device.htod_copy_into(host_a.to_vec(), &mut a).unwrap();
    device.htod_copy_into(host_b.to_vec(), &mut b).unwrap();

    for f in [
        "masked_xor_256",
        "masked_xor_256_x2",
        "masked_xor_256_x4",
        "masked_xor_256_x8",
    ] {
        let func = device.get_func("", f).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (NUM_ELEMENTS.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 32,
        };

        let start = std::time::Instant::now();
        unsafe {
            func.clone()
                .launch(cfg, (&mut a, &mut b, &mut c, NUM_ELEMENTS as u32))
        }
        .unwrap();
        device.synchronize().unwrap();
        let end = std::time::Instant::now();
        println!("{}: {:?}", f, end - start);
    }
}
