use cudarc::{
    driver::{
        CudaDevice, CudaFunction, CudaSlice, CudaStream, CudaViewMut, DeviceSlice, LaunchAsync,
        LaunchConfig,
    },
    nvrtc::compile_ptx,
};
use std::sync::Arc;

pub(super) struct ChachaCommon {
    /// the current state of the chacha rng
    pub(super) chacha_ctx: ChaChaCtx,
    // the current state of the chacha rng on the device
    pub(super) state_gpu_buf: CudaSlice<u32>,
}

impl ChachaCommon {
    pub const CHACHA_PTX_SRC: &str = include_str!("chacha.cu");
    pub const CHACHA_FILL_FUNCTION_NAME: &str = "chacha12";
    pub const CHACHA_XOR_FUNCTION_NAME: &str = "chacha12_xor";

    pub fn init(dev: &Arc<CudaDevice>, seed: [u32; 8]) -> Self {
        let chacha_ctx = ChaChaCtx::init(seed, 0, 0);
        let state_gpu_buf = dev.htod_sync_copy(chacha_ctx.state.as_ref()).unwrap();

        Self {
            chacha_ctx,
            state_gpu_buf,
        }
    }

    pub(super) fn advance_counter(&mut self, num_ks_calls: u64) {
        let mut counter = self.chacha_ctx.get_counter();
        counter += num_ks_calls;
        self.chacha_ctx.set_counter(counter);
    }

    pub fn launch_config_from_elements_and_threads(
        num_elements: u32,
        threads: u32,
        device: &Arc<CudaDevice>,
    ) -> LaunchConfig {
        let num_blocks = num_elements.div_ceil(threads);
        LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub fn fill_rng_into(
        &mut self,
        buf: &mut CudaViewMut<u32>,
        stream: &CudaStream,
        fill_kernel: &CudaFunction,
    ) {
        let len = buf.len();
        assert!(len % 16 == 0, "buffer length must be a multiple of 16");
        let num_ks_calls = len / 16; // we produce 16 u32s per kernel call
        let threads_per_block = 256; // ON CHANGE: sync with kernel

        let cfg = Self::launch_config_from_elements_and_threads(
            num_ks_calls as u32,
            threads_per_block,
            &self.state_gpu_buf.device(),
        );

        unsafe {
            fill_kernel
                .clone()
                .launch_on_stream(
                    stream,
                    cfg,
                    (
                        &mut *buf,
                        &self.state_gpu_buf,
                        self.chacha_ctx.state[12], // first part of counter
                        self.chacha_ctx.state[13], // second part of counter
                        len,
                    ),
                )
                .unwrap();
        }
        // increment the state counter of the ChaChaRng with the number of produced
        // blocks
        // one call to KS produces 16 u32, so we increase the counter by the number of
        // KS calls
        self.advance_counter(num_ks_calls as u64);
    }
}

pub struct ChaChaCudaRng {
    dev: Arc<CudaDevice>,
    fill_kernel: CudaFunction,
    rng_chunk: Option<CudaSlice<u32>>,
    output_buffer: Option<Vec<u32>>,
    chacha: ChachaCommon,
}

impl ChaChaCudaRng {
    // takes number of bytes to produce, buffer has u32 datatype so will produce
    // buf_size/4 u32s
    pub fn init(buf_size_bytes: usize, dev: Arc<CudaDevice>, seed: [u32; 8]) -> Self {
        let ptx = compile_ptx(ChachaCommon::CHACHA_PTX_SRC).unwrap();

        assert!(
            buf_size_bytes % 64 == 0,
            "buf_size must be a multiple of 64 atm"
        );

        dev.load_ptx(
            ptx.clone(),
            ChachaCommon::CHACHA_FILL_FUNCTION_NAME,
            &[ChachaCommon::CHACHA_FILL_FUNCTION_NAME],
        )
        .unwrap();
        let fill_kernel = dev
            .get_func(
                ChachaCommon::CHACHA_FILL_FUNCTION_NAME,
                ChachaCommon::CHACHA_FILL_FUNCTION_NAME,
            )
            .unwrap();

        let chacha = ChachaCommon::init(&dev, seed);

        if buf_size_bytes == 0 {
            return Self {
                dev,
                fill_kernel,
                rng_chunk: None,
                output_buffer: None,
                chacha,
            };
        }
        let buf = vec![0u32; buf_size_bytes / 4];
        let rng_chunk = dev.htod_sync_copy(&buf).unwrap();

        Self {
            dev,
            fill_kernel,
            rng_chunk: Some(rng_chunk),
            output_buffer: Some(buf),
            chacha,
        }
    }
    pub fn init_empty(dev: Arc<CudaDevice>, seed: [u32; 8]) -> Self {
        Self::init(0, dev, seed)
    }

    pub fn fill_rng(&mut self) {
        assert!(self.rng_chunk.is_some() && self.output_buffer.is_some());

        self.fill_rng_no_host_copy(
            self.output_buffer.as_ref().unwrap().len() * std::mem::size_of::<u32>(),
            &self.dev.fork_default_stream().unwrap(),
        );

        self.dev
            .dtoh_sync_copy_into(
                self.rng_chunk.as_ref().unwrap(),
                self.output_buffer.as_mut().unwrap(),
            )
            .unwrap();
    }

    pub fn fill_rng_no_host_copy(&mut self, buf_size_bytes: usize, stream: &CudaStream) {
        assert!(self.rng_chunk.is_some());

        let mut buf = self
            .rng_chunk
            .as_mut()
            .unwrap()
            .slice_mut(0..buf_size_bytes / 4);

        self.chacha
            .fill_rng_into(&mut buf, stream, &self.fill_kernel);
    }

    pub fn fill_rng_into(&mut self, buf: &mut CudaViewMut<u32>, stream: &CudaStream) {
        self.chacha.fill_rng_into(buf, stream, &self.fill_kernel);
    }

    pub fn data(&self) -> Option<&[u32]> {
        self.output_buffer.as_deref()
    }

    pub fn get_mut_chacha(&mut self) -> &mut ChaChaCtx {
        &mut self.chacha.chacha_ctx
    }

    pub fn cuda_slice(&self) -> Option<&CudaSlice<u32>> {
        self.rng_chunk.as_ref()
    }
    pub fn set_cuda_slice(&mut self, slice: CudaSlice<u32>) {
        assert!(self.rng_chunk.is_none());
        assert!(
            slice.len() % 16 == 0,
            "slice length must be a multiple of 16"
        );
        self.rng_chunk = Some(slice);
    }
    pub fn take_cuda_slice(&mut self) -> CudaSlice<u32> {
        self.rng_chunk.take().unwrap()
    }
}

// Modeled after:
// struct chacha_ctx
// {
//     uint32_t keystream[16];
//     uint32_t state[16];
//     uint32_t *counter;
// };

pub struct ChaChaCtx {
    // 12 32-bit words for the key
    // 2 32-bit words for the counter
    // 2 32-bit words for the nonce (stream id)
    pub(crate) state: [u32; 16],
}

const CHACONST: [u32; 4] = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574];

impl ChaChaCtx {
    const COUNTER_START_IDX: usize = 12;
    const NONCE_START_IDX: usize = 14;
    pub fn init(key: [u32; 8], counter: u64, nonce: u64) -> Self {
        let mut state = [0u32; 16];
        state[0] = CHACONST[0];
        state[1] = CHACONST[1];
        state[2] = CHACONST[2];
        state[3] = CHACONST[3];
        state[4..12].copy_from_slice(&key);

        let mut res = Self { state };
        res.set_counter(counter);
        res.set_nonce(nonce);
        res
    }
    fn get_value(&self, idx: usize) -> u64 {
        self.state[idx] as u64 | ((self.state[idx + 1] as u64) << 32)
    }
    fn set_value(&mut self, idx: usize, value: u64) {
        self.state[idx] = value as u32;
        self.state[idx + 1] = (value >> 32) as u32;
    }

    pub fn set_counter(&mut self, counter: u64) {
        self.set_value(Self::COUNTER_START_IDX, counter)
    }
    pub fn get_counter(&self) -> u64 {
        self.get_value(Self::COUNTER_START_IDX)
    }
    pub fn set_nonce(&mut self, nonce: u64) {
        self.set_value(Self::NONCE_START_IDX, nonce)
    }
    pub fn get_nonce(&self) -> u64 {
        self.get_value(Self::NONCE_START_IDX)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_chacha_rng() {
        // This call to CudaDevice::new is only used in context of a test - not used in
        // the server binary
        let mut rng = ChaChaCudaRng::init(1024 * 1024, CudaDevice::new(0).unwrap(), [0u32; 8]);
        rng.fill_rng();
        let zeros = rng.data().unwrap().iter().filter(|x| x == &&0).count();
        // we would expect no 0s in the output buffer even 1 is 1/4096;
        assert!(zeros <= 1);
        let data = rng.data().unwrap().to_vec();
        rng.fill_rng();
        assert!(&data[..] != rng.data().unwrap());
    }
}
