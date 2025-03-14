using PhyloGaussianBeliefProp
using Documenter
using DocumenterCitations

using DocumenterInterLinks
links = InterLinks(
    "PhyloNetworks" => "https://juliaphylo.github.io/PhyloNetworks.jl/stable/objects.inv",
    "PhyloTraits" => "https://juliaphylo.github.io/PhyloTraits.jl/stable/objects.inv",
)
DocMeta.setdocmeta!(PhyloGaussianBeliefProp, :DocTestSetup, :(using PhyloGaussianBeliefProp); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    modules=[PhyloGaussianBeliefProp],
    authors="Cecile Ane <cecileane@users.noreply.github.com>, Benjamin Teo <bstkj@users.noreply.github.com>, and contributors",
    sitename="PhyloGaussianBeliefProp.jl",
    format=Documenter.HTML(;
        mathengine=Documenter.KaTeX(),
        prettyurls=get(ENV, "CI", "false") == "true",
        size_threshold = 600 * 2^10, size_threshold_warn = 500 * 2^10, # 600 KiB
        canonical="https://JuliaPhylo.github.io/PhyloGaussianBeliefProp.jl/stable/",
        edit_link="main",
        assets=String["assets/citations.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Installation" => "man/installation.md",
            "Getting started" => "man/getting_started.md",
            "Background" => "man/background.md",
            "Evolutionary models" => "man/evolutionary_models.md",
            "Cluster graphs" => "man/clustergraphs.md",
            "Regularization" => "man/regularization.md",
            "Message schedules" => "man/message_schedules.md"
        ]
    ],
    doctestfilters=[
        # Ignore any digit after the 5th digit after a decimal, throughout the docs
        r"(?<=\d\.\d{5})\d+",
    ],
    plugins=[bib,links],
)

deploydocs(;
    repo="github.com/JuliaPhylo/PhyloGaussianBeliefProp.jl.git",
    push_preview=true,
    devbranch="main",
)
