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
    return write(path, content)
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
Inject Google Search Console verification meta tag into generated HTML files.
This is enabled only when `ENV["GOOGLE_SITE_VERIFICATION"]` is set.
"""
function inject_google_site_verification!(build_dir::String)
    token = strip(get(ENV, "GOOGLE_SITE_VERIFICATION", ""))
    isempty(token) && return

    safe_token = replace(token, '"' => "&quot;")
    meta_tag = "<meta name=\"google-site-verification\" content=\"$(safe_token)\" />"
    injected = 0

    for (root, _, files) in walkdir(build_dir)
        for file in files
            endswith(file, ".html") || continue
            path = joinpath(root, file)
            html = read(path, String)
            occursin("google-site-verification", html) && continue
            occursin("</head>", html) || continue

            write_if_changed(path, replace(html, "</head>" => "$(meta_tag)\n</head>"; count = 1))
            injected += 1
        end
    end

    return @info "Injected google-site-verification meta tag" files = injected build_dir = build_dir
end

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
    # servedocs() sets root to docs/ which conflicts with project-root remotes.
    # Enable GitHub source links only in CI where makedocs root matches git root.
    remotes = get(ENV, "CI", nothing) == "true" ?
        Dict(dirname(@__DIR__) => (Documenter.Remotes.GitHub("ProjectTorreyPines", "AdaptiveArrayPools.jl"), "master")) :
        nothing,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://projecttorreypines.github.io/AdaptiveArrayPools.jl",
        edit_link = :commit,
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Basics" => [
            "Quick Start" => "basics/quick-start.md",
            "`@with_pool` Patterns" => "basics/with-pool-patterns.md",
            "Essential API" => "basics/api-essentials.md",
            "Safety Rules" => "basics/safety-rules.md",
        ],
        "Advanced" => [
            "Pool Patterns" => "advanced/pool-patterns.md",
            "Multi-threading" => "features/multi-threading.md",
        ],
        "Features" => [
            "Pool Safety" => "features/safety.md",
            "`@maybe_with_pool`" => "features/maybe-with-pool.md",
            "Bit Arrays" => "features/bit-arrays.md",
            "CUDA Support" => "features/cuda-support.md",
            "Configuration" => "features/configuration.md",
        ],
        "Reference" => [
            "Full API" => "reference/api.md",
        ],
        "Architecture" => [
            "How It Works" => "architecture/how-it-works.md",
            "Type Dispatch & Cache" => "architecture/type-dispatch.md",
            "`@with_pool` Internals" => "architecture/macro-internals.md",
            "Design Documents" => "architecture/design-docs.md",
        ],
    ],
    doctest = true,
    checkdocs = :exports,
    warnonly = [:cross_references, :missing_docs],
)

inject_google_site_verification!(joinpath(@__DIR__, "build"))

deploydocs(
    repo = "github.com/ProjectTorreyPines/AdaptiveArrayPools.jl.git",
    devbranch = "master",
    push_preview = true,
)
