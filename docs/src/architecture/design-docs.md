# Design Documents

The historical design documents that used to live under `docs/design/` (covering
the hybrid API strategy, the N-D array approach comparison, fixed-slot codegen,
untracked-acquire detection, and the CUDA extension) have been removed as
outdated — the decisions they recorded either shipped and became the current
implementation, or were superseded by later approaches (e.g. the N-way cache
described in some of them is legacy on Julia 1.12+, see below).

Current architecture documentation lives in [How It Works](how-it-works.md),
which explains the mechanisms that enable zero-allocation array reuse as they
exist today.

## See Also

- [How It Works](how-it-works.md) - High-level architecture overview
- [Type Dispatch & Cache](type-dispatch.md) - Technical deep-dive
- [@with_pool Macro Internals](macro-internals.md) - Macro transformation details
