# Metal `trim!` memory smoke test (Apple Silicon).
#
# Not a unit test — a manual script that prints memory before/after `trim!` to
# show that reclamation actually happens. Run in an env with Metal + AdaptiveArrayPools:
#
#   julia --project=<env> scripts/trim_smoke_metal.jl
#
# Three probes:
#   julia_live  = Base.gc_live_bytes()                  Julia-tracked live heap
#   rss         = process RSS (ps)                       OS resident memory
#   metal_alloc = Metal.device().currentAllocatedSize    Metal device allocation
#                 (Apple Silicon is unified-memory, so this is shared RAM)
#
# Metal has no reclaim() (no caching allocator exposed); force_gc=true runs
# GC.gc(), whose finalizers free the MtlArray buffers.

using Metal
using AdaptiveArrayPools

human(b) = Base.format_bytes(b)
jlive() = Base.gc_live_bytes()
rss() =
try
    parse(Int, readchomp(`ps -o rss= -p $(getpid())`)) * 1024
catch
    -1
end
metal_alloc() = Int(Metal.device().currentAllocatedSize)

function report(label)
    return println(
        rpad(label, 30),
        "julia_live=", rpad(human(jlive()), 11),
        "  rss=", rpad(rss() < 0 ? "?" : human(rss()), 11),
        "  metal_alloc=", human(metal_alloc()),
    )
end

const GiB = 1024^3
const N = GiB ÷ sizeof(Float32)   # 1 GiB of Float32 per slot
const NSLOTS = 4                  # 4 GiB total

pool = get_task_local_metal_pool()
GC.gc(true)
report("0) start")

# Allocate NSLOTS x 1 GiB into the pool (fill! to commit), inside a scope.
checkpoint!(pool)
for i in 1:NSLOTS
    v = acquire!(pool, Float32, N)
    fill!(v, Float32(i))
end
Metal.synchronize()
report("1) allocated $(NSLOTS) GiB")

rewind!(pool)                     # slots now inactive (retained for reuse)
report("2) after rewind! (retained)")

summary = trim!(pool; force_gc = true)   # detach inactive buffers + GC
GC.gc(true)
report("3) after trim!(force_gc=true)")
println("    summary = ", summary)
