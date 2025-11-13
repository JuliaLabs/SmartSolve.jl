using SmartSolve
using Documenter

DocMeta.setdocmeta!(SmartSolve, :DocTestSetup, :(using SmartSolve); recursive=true)

makedocs(;
    modules=[SmartSolve],
    authors="JuliaLabs",
    sitename="SmartSolve.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaLabs.github.io/SmartSolve.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaLabs/SmartSolve.jl",
    devbranch="main",
)
