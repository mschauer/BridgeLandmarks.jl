using Documenter, BridgeLandmarks

makedocs(;
    modules=[BridgeLandmarks],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/mschauer/BridgeLandmarks.jl/blob/{commit}{path}#L{line}",
    sitename="BridgeLandmarks.jl",
    authors="mschauer <moritzschauer@web.de>",
    assets=String[],
)

deploydocs(;
    repo="github.com/mschauer/BridgeLandmarks.jl",
)
