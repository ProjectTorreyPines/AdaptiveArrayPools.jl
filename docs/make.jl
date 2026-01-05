using Documenter
using AdaptiveArrayPools

# ============================================
# Helper: Conditional write (for LiveServer compatibility)
# ============================================

"""
Write file only if content changed (prevents LiveServer infinite loop).
"""
function write_if_changed(path::String, content::String)
    if isfile(path) && read(path, String) == content
        return  # Content unchanged, skip write
    end
    write(path, content)
end

# ============================================
# Helper: Rewrite relative paths in README
# ============================================

const GITHUB_PAGES_BASE = "https://projecttorreypines.github.io/AdaptiveArrayPools.jl/stable"
const REPO_URL = "https://github.com/ProjectTorreyPines/AdaptiveArrayPools.jl"

"""
Rewrite relative paths in README.md for Documenter structure.

Converts GitHub repo links to internal Documenter links:
- `docs/api.md` → `usage/api.md`
- `docs/cuda.md` → `usage/cuda.md`
- `docs/safety.md` → `guide/safety.md`
- `docs/multi-threading.md` → `advanced/multi-threading.md`
- `docs/configuration.md` → `usage/configuration.md`
- `docs/maybe_with_pool.md` → `usage/maybe_with_pool.md`

Also handles anchor links (e.g., `docs/api.md#convenience-functions`).
"""
function rewrite_readme_paths(content::String)
    # Usage docs (with optional anchors)
    content = replace(content, r"\(docs/api\.md(#[^)]+)?\)" => s"(usage/api.md\1)")
    content = replace(content, r"\(docs/cuda\.md(#[^)]+)?\)" => s"(usage/cuda.md\1)")
    content = replace(content, r"\(docs/configuration\.md(#[^)]+)?\)" => s"(usage/configuration.md\1)")
    content = replace(content, r"\(docs/maybe_with_pool\.md(#[^)]+)?\)" => s"(usage/maybe_with_pool.md\1)")

    # Guide docs
    content = replace(content, r"\(docs/safety\.md(#[^)]+)?\)" => s"(guide/safety.md\1)")

    # Advanced docs
    content = replace(content, r"\(docs/multi-threading\.md(#[^)]+)?\)" => s"(advanced/multi-threading.md\1)")

    # LICENSE link → GitHub
    content = replace(content, "(LICENSE)" => "($(REPO_URL)/blob/master/LICENSE)")

    return content
end

# ============================================
# Generate index.md from README
# ============================================

const DOCS_DIR = @__DIR__
const DOCS_SRC = joinpath(DOCS_DIR, "src")

# README.md → index.md (with path rewriting)
readme_content = read(joinpath(DOCS_DIR, "../README.md"), String)
write_if_changed(joinpath(DOCS_SRC, "index.md"), rewrite_readme_paths(readme_content))

# ============================================
# Build documentation
# ============================================

makedocs(
    sitename = "AdaptiveArrayPools.jl",
    authors = "Min-Gu Yoo",
    modules = [AdaptiveArrayPools],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://projecttorreypines.github.io/AdaptiveArrayPools.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Getting Started" => "guide/getting-started.md",
            "Safety Rules" => "guide/safety.md",
        ],
        "Usage" => [
            "API Reference" => "usage/api.md",
            "Configuration" => "usage/configuration.md",
            "@maybe_with_pool" => "usage/maybe_with_pool.md",
            "CUDA Support" => "usage/cuda.md",
        ],
        "Advanced" => [
            "Multi-threading" => "advanced/multi-threading.md",
            "How @with_pool Works" => "advanced/macro-internals.md",
            "Internals" => "advanced/internals.md",
        ],
    ],
    doctest = false,  # Doctests not set up in existing docs
    checkdocs = :none,  # Using manual API tables, not @autodocs
    warnonly = [:cross_references, :missing_docs],
)

deploydocs(
    repo = "github.com/ProjectTorreyPines/AdaptiveArrayPools.jl.git",
    devbranch = "master",
    push_preview = false,  # Deploy only on master/tag, not on PR
)
