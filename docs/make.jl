using PhyloGaussianBeliefProp
using Documenter

DocMeta.setdocmeta!(PhyloGaussianBeliefProp, :DocTestSetup, :(using PhyloGaussianBeliefProp); recursive=true)

makedocs(;
    modules=[PhyloGaussianBeliefProp],
    authors="Cecile Ane <cecileane@users.noreply.github.com>, Benjamin Teo <bstkj@users.noreply.github.com>, and contributors",
    repo="https://github.com/cecileane/PhyloGaussianBeliefProp.jl/blob/{commit}{path}#{line}",
    sitename="PhyloGaussianBeliefProp.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cecileane.github.io/PhyloGaussianBeliefProp.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cecileane/PhyloGaussianBeliefProp.jl",
    devbranch="main",
)
