# Design Documents

For in-depth analysis of design decisions, implementation tradeoffs, and architectural choices, see the design documents in the repository:

## API Design

- **[hybrid_api_design.md](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/hybrid_api_design.md)**
  Two-API strategy (`acquire!` vs `unsafe_acquire!`) and type stability analysis

## Caching & Performance

- **[nd_array_approach_comparison.md](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/nd_array_approach_comparison.md)**
  N-way cache design, boxing analysis, and ReshapedArray benchmarks

- **[fixed_slots_codegen_design.md](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/fixed_slots_codegen_design.md)**
  Zero-allocation iteration via `@generated` functions and fixed-slot type dispatch

## Macro Internals

- **[untracked_acquire_design.md](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/untracked_acquire_design.md)**
  Macro-based untracked acquire detection and 1-based sentinel pattern

## Backend Extensions

- **[cuda_extension_design.md](https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl/blob/master/docs/design/cuda_extension_design.md)**
  CUDA backend architecture and package extension loading

---

## Document Overview

| Document | Focus Area | Key Insights |
|----------|------------|--------------|
| hybrid_api_design | API strategy | View types for zero-alloc, Array for FFI |
| nd_array_approach_comparison | Caching | N-way associative cache reduces header allocation |
| fixed_slots_codegen_design | Codegen | @generated functions enable type-stable iteration |
| untracked_acquire_design | Macro safety | Sentinel pattern ensures correct cleanup |
| cuda_extension_design | GPU support | Seamless CPU/CUDA API parity |

## See Also

- [How It Works](how-it-works.md) - High-level architecture overview
- [Type Dispatch & Cache](type-dispatch.md) - Technical deep-dive
- [@with_pool Macro Internals](macro-internals.md) - Macro transformation details
