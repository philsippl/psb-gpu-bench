template<int NUM_LIMBS>
struct BigInt {
    unsigned long long limbs[NUM_LIMBS];

    __device__ BigInt() {
        for (int i = 0; i < NUM_LIMBS; i++) {
            limbs[i] = 0;
        }
    }

    __device__ BigInt(const unsigned long long* input_limbs) {
        for (int i = 0; i < NUM_LIMBS; i++) {
            limbs[i] = input_limbs[i];
        }
    }

    __device__ BigInt operator^(const BigInt& other) const {
        BigInt result;
        for (int i = 0; i < NUM_LIMBS; i++) {
            result.limbs[i] = this->limbs[i] ^ other.limbs[i];
        }
        return result;
    }

    __device__ BigInt operator&(const BigInt& other) const {
        BigInt result;
        for (int i = 0; i < NUM_LIMBS; i++) {
            result.limbs[i] = this->limbs[i] & other.limbs[i];
        }
        return result;
    }

    __device__ void atomic_xor_into(BigInt& target) const {
        for (int i = 0; i < NUM_LIMBS; i++) {
            atomicXor(&target.limbs[i], this->limbs[i]);
        }
    }
};

template<int NUM_LIMBS>
__global__ void masked_xor(BigInt<NUM_LIMBS>* a, BigInt<NUM_LIMBS>* b, BigInt<NUM_LIMBS>* c, unsigned int num_elements) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    BigInt<NUM_LIMBS> local_result;
    
    if (idx < num_elements) {
        BigInt<NUM_LIMBS> and_result = a[idx] & b[idx];
        local_result = local_result ^ and_result;
    }
    
    extern __shared__ unsigned char shared_mem[];
    BigInt<NUM_LIMBS>* shared_data = reinterpret_cast<BigInt<NUM_LIMBS>*>(shared_mem);
    shared_data[threadIdx.x] = local_result;
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] = shared_data[threadIdx.x] ^ shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        shared_data[0].atomic_xor_into(c[0]);
    }
}

extern "C" __global__ void masked_xor_64(BigInt<1>* a, BigInt<1>* b, BigInt<1>* c, unsigned int num_elements) {
    masked_xor<1>(a, b, c, num_elements);
}

extern "C" __global__ void masked_xor_128(BigInt<2>* a, BigInt<2>* b, BigInt<2>* c, unsigned int num_elements) {
    masked_xor<2>(a, b, c, num_elements);
}

extern "C" __global__ void masked_xor_256(BigInt<4>* a, BigInt<4>* b, BigInt<4>* c, unsigned int num_elements) {
    masked_xor<4>(a, b, c, num_elements);
}
