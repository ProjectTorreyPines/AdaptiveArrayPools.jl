# ==============================================================================
# Metal Auto-Compact: service a flagged Metal pool at its `@with_pool :metal` entry.
# ==============================================================================
#
# Parity with the CPU `_maybe_auto_manage!(::AdaptiveArrayPool)` in src/auto_manage.jl.
# The single global Timer (base module, CPU thread) sets this pool's `@atomic
# _compact_requested` flag via the shared WeakRef registry; the owner task runs the full
# `compact!` here, at the `_current_depth == 1` safepoint where nothing is borrowed.
#
# The scope-entry hook itself is already emitted by the SHARED `@with_pool` codegen
# (it expands `@with_pool :metal` through the same `_generate_block_inner`), so this
# concrete method is all that's needed to make the hook act on Metal pools — before it,
# the hook resolved to the `::Any` no-op fallback. Reuses the base `_run_auto_manage!`
# (Metal `compact!` takes the same factor/shrink_to/min_bytes/active kwargs as CPU).
#
# `@inline` so the common case (flag clear) is a cheap inlined monotonic read + compare;
# the cold `_run_auto_manage!` stays a non-inlined call. `:monotonic` matches CPU: the
# flag is a one-way eventual-visibility signal with no cross-field ordering invariant.

@inline function AdaptiveArrayPools._maybe_auto_manage!(pool::MetalAdaptiveArrayPool)
    if (@atomic :monotonic pool._compact_requested) && pool._current_depth == 1
        AdaptiveArrayPools._run_auto_manage!(pool)
    end
    return nothing
end
