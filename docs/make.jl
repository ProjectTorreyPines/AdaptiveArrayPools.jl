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

# Path mapping table: (pattern, replacement)
# Order matters for overlapping patterns
const README_PATH_MAPPINGS = [
    # Reference
    (r"\(docs/api\.md(#[^)]+)?\)", s"(reference/api.md\1)"),

    # Features
    (r"\(docs/cuda\.md(#[^)]+)?\)", s"(features/cuda-support.md\1)"),
    (r"\(docs/configuration\.md(#[^)]+)?\)", s"(features/configuration.md\1)"),
    (r"\(docs/maybe_with_pool\.md(#[^)]+)?\)", s"(features/maybe-with-pool.md\1)"),
    (r"\(docs/multi-threading\.md(#[^)]+)?\)", s"(features/multi-threading.md\1)"),

    # Basics
    (r"\(docs/safety\.md(#[^)]+)?\)", s"(basics/safety-rules.md\1)"),
]

"""
Rewrite relative paths in README.md for Documenter structure.

Uses mapping table to convert GitHub repo links to internal Documenter links.
Also handles anchor links (e.g., `docs/api.md#convenience-functions`).
"""
function rewrite_readme_paths(content::String)
    for (pattern, replacement) in README_PATH_MAPPINGS
        content = replace(content, pattern => replacement)
    end

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
        "Basics" => [
            "Quick Start" => "basics/quick-start.md",
            "@with_pool Patterns" => "basics/with-pool-patterns.md",
            "Essential API" => "basics/api-essentials.md",
            "Safety Rules" => "basics/safety-rules.md",
        ],
        "Features" => [
            "@maybe_with_pool" => "features/maybe-with-pool.md",
            "CUDA Support" => "features/cuda-support.md",
            "Multi-threading" => "features/multi-threading.md",
            "Configuration" => "features/configuration.md",
        ],
        "Reference" => [
            "Full API" => "reference/api.md",
        ],
        "Architecture" => [
            "How It Works" => "architecture/how-it-works.md",
            "Type Dispatch & Cache" => "architecture/type-dispatch.md",
            "@with_pool Internals" => "architecture/macro-internals.md",
            "Design Documents" => "architecture/design-docs.md",
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
