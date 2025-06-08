use std::sync::Arc;

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};

const THREAD_COUNT: usize = 256;

pub struct MaskedXor {
    limbs: usize,
    device: Arc<CudaDevice>,
    func: CudaFunction,
}

impl MaskedXor {
    pub fn new(limbs: usize) -> Self {
        let device = CudaDevice::new(0).unwrap();
        let ptx = compile_ptx(include_str!("kernel.cu")).unwrap();
        let f = match limbs {
            1 => "masked_xor_64",
            2 => "masked_xor_128",
            4 => "masked_xor_256",
            _ => unimplemented!(),
        };
        device.load_ptx(ptx.clone(), "", &[f]).unwrap();
        let func = device.get_func("", f).unwrap();
        Self {
            limbs,
            device,
            func,
        }
    }

    pub fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }

    pub fn run(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        assert_eq!(a.len(), b.len());
        let a = self.device.htod_copy(a.to_vec()).unwrap();
        let b = self.device.htod_copy(b.to_vec()).unwrap();

        self.run_with_device_ptrs(&a, &b)
    }

    pub fn run_with_device_ptrs(&self, a: &CudaSlice<u64>, b: &CudaSlice<u64>) -> Vec<u64> {
        let num_elements: usize = a.len() / self.limbs;
        let cfg = LaunchConfig {
            grid_dim: (num_elements.div_ceil(THREAD_COUNT) as u32, 1, 1),
            block_dim: (THREAD_COUNT as u32, 1, 1),
            shared_mem_bytes: THREAD_COUNT as u32 * 32,
        };

        let c = self.device.alloc_zeros(self.limbs).unwrap();

        unsafe {
            self.func
                .clone()
                .launch(cfg, (a, b, &c, num_elements as u32))
                .unwrap();
        }

        self.device.dtoh_sync_copy(&c).unwrap()
    }
}

mod tests {
    use super::*;

    fn ref_masked_xor(a: &[u64], b: &[u64], c: &mut [u64]) {
        for (a_, b_) in a.chunks(4).zip(b.chunks(4)) {
            c[0] ^= a_[0] & b_[0];
            c[1] ^= a_[1] & b_[1];
            c[2] ^= a_[2] & b_[2];
            c[3] ^= a_[3] & b_[3];
        }
    }

    #[test]
    fn test_against_reference() {
        const LIMBS: usize = 4;
        const BUFFER_SIZE: usize = 1024;
        let host_a: Vec<u64> = (0..BUFFER_SIZE).map(|_| rand::random::<u64>()).collect();
        let host_b: Vec<u64> = (0..BUFFER_SIZE).map(|_| rand::random::<u64>()).collect();
        let mut ref_c = [0u64; LIMBS];
        ref_masked_xor(&host_a, &host_b, &mut ref_c);
        let instance = MaskedXor::new(LIMBS);
        assert_eq!(instance.run(&host_a, &host_b), ref_c);
    }
}
