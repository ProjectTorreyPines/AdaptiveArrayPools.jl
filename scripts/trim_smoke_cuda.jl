# CUDA `trim!` memory smoke test (run on an NVIDIA GPU machine).
#
# Not a unit test — a manual script that prints memory before/after `trim!` to
# show whether reclamation actually happens. Run in an env with CUDA + AdaptiveArrayPools:
#
#   julia --project=<env> scripts/trim_smoke_cuda.jl
#
# Probes:
#   julia_live = Base.gc_live_bytes()                        Julia-tracked live heap
#   rss        = process RSS (ps)                             OS resident memory
#   vram_used  = CUDA.total_memory() - CUDA.available_memory()  device VRAM in use
#
# On CUDA, trim!(force_gc=true) additionally calls CUDA.reclaim() to return
# pooled blocks to the driver. NOTE: a Metal-specific wrapper double-copy bug
# (since fixed) had prevented Metal from reclaiming. CUDA's CuArray constructor
# takes ownership of its DataRef (no internal copy), so CUDA does not have that
# bug and is expected to reclaim here. This script confirms it on real hardware:
# vram_used should fall at step 3.

using CUDA
using AdaptiveArrayPools

human(b) = Base.format_bytes(b)
jlive() = Base.gc_live_bytes()
rss() =
try
    parse(Int, readchomp(`ps -o rss= -p $(getpid())`)) * 1024
catch
    -1
end
vram_used() = Int(CUDA.total_memory() - CUDA.available_memory())

function report(label)
    return println(
        rpad(label, 30),
        "julia_live=", rpad(human(jlive()), 11),
        "  rss=", rpad(rss() < 0 ? "?" : human(rss()), 11),
        "  vram_used=", human(vram_used()),
    )
end

const GiB = 1024^3
const N = GiB ÷ sizeof(Float32)   # 1 GiB of Float32 per slot
const NSLOTS = 4                  # 4 GiB total

pool = get_task_local_cuda_pool()
GC.gc(true); CUDA.reclaim()
report("0) start")

checkpoint!(pool)
for i in 1:NSLOTS
    v = acquire!(pool, Float32, N)
    fill!(v, Float32(i))
end
CUDA.synchronize()
report("1) allocated $(NSLOTS) GiB")

rewind!(pool)                     # slots now inactive (retained for reuse)
report("2) after rewind! (retained)")

summary = trim!(pool; force_gc = true)   # detach inactive buffers + GC.gc() + CUDA.reclaim()
GC.gc(true)
report("3) after trim!(force_gc=true)")
println("    summary = ", summary)
println()
CUDA.memory_status()              # detailed pool/driver VRAM breakdown
