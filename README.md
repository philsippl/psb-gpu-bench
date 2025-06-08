# Results

## A100

```
     Running benches/chacha_bench.rs (target/release/deps/chacha_bench-cb9cf92af5b37634)
Timer precision: 20 ns
chacha_bench      fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ chacha_rng                   │               │               │               │         │
   ├─ 1024        72.74 ms      │ 74.78 ms      │ 73.29 ms      │ 73.44 ms      │ 10      │ 10
   │              14.07 Kitem/s │ 13.69 Kitem/s │ 13.97 Kitem/s │ 13.94 Kitem/s │         │
   ├─ 1048576     73.19 ms      │ 75.37 ms      │ 73.55 ms      │ 73.79 ms      │ 10      │ 10
   │              14.32 Mitem/s │ 13.91 Mitem/s │ 14.25 Mitem/s │ 14.2 Mitem/s  │         │
   ╰─ 1073741824  197.6 ms      │ 200.2 ms      │ 198.3 ms      │ 198.6 ms      │ 10      │ 10
                  5.433 Gitem/s │ 5.362 Gitem/s │ 5.412 Gitem/s │ 5.404 Gitem/s │         │

     Running benches/masked_xor_bench.rs (target/release/deps/masked_xor_bench-4927de7fc1742718)
Timer precision: 20 ns
masked_xor_bench   fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ masked_xor_64                 │               │               │               │         │
│  ├─ 1024         20.09 µs      │ 40.12 µs      │ 20.68 µs      │ 22.83 µs      │ 10      │ 10
│  │               101.8 Mitem/s │ 51.03 Mitem/s │ 99 Mitem/s    │ 89.67 Mitem/s │         │
│  ├─ 1048576      41.36 µs      │ 162 µs        │ 48.47 µs      │ 58.96 µs      │ 10      │ 10
│  │               50.7 Gitem/s  │ 12.94 Gitem/s │ 43.26 Gitem/s │ 35.56 Gitem/s │         │
│  ╰─ 16777216     356.1 µs      │ 493.8 µs      │ 492.8 µs      │ 479 µs        │ 10      │ 10
│                  94.2 Gitem/s  │ 67.94 Gitem/s │ 68.08 Gitem/s │ 70.04 Gitem/s │         │
├─ masked_xor_128                │               │               │               │         │
│  ├─ 1024         20.23 µs      │ 41.24 µs      │ 20.57 µs      │ 22.97 µs      │ 10      │ 10
│  │               101.2 Mitem/s │ 49.65 Mitem/s │ 99.53 Mitem/s │ 89.15 Mitem/s │         │
│  ├─ 1048576      70.81 µs      │ 132.4 µs      │ 76.64 µs      │ 81.92 µs      │ 10      │ 10
│  │               29.61 Gitem/s │ 15.83 Gitem/s │ 27.36 Gitem/s │ 25.59 Gitem/s │         │
│  ╰─ 16777216     668.1 µs      │ 975.8 µs      │ 961.9 µs      │ 933.3 µs      │ 10      │ 10
│                  50.22 Gitem/s │ 34.38 Gitem/s │ 34.88 Gitem/s │ 35.94 Gitem/s │         │
╰─ masked_xor_256                │               │               │               │         │
   ├─ 1024         20.38 µs      │ 44.28 µs      │ 20.96 µs      │ 23.36 µs      │ 10      │ 10
   │               100.4 Mitem/s │ 46.24 Mitem/s │ 97.67 Mitem/s │ 87.64 Mitem/s │         │
   ├─ 1048576      139.4 µs      │ 161.1 µs      │ 143.1 µs      │ 144.4 µs      │ 10      │ 10
   │               15.03 Gitem/s │ 13.01 Gitem/s │ 14.64 Gitem/s │ 14.52 Gitem/s │         │
   ╰─ 16777216     1.236 ms      │ 1.857 ms      │ 1.847 ms      │ 1.786 ms      │ 10      │ 10
                   27.14 Gitem/s │ 18.06 Gitem/s │ 18.16 Gitem/s │ 18.77 Gitem/s │         │
```