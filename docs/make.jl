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
        size_threshold = 600 * 2^10, size_threshold_warn = 500 * 2^10, # 600 KiB
        canonical="https://cecileane.github.io/PhyloGaussianBeliefProp.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Installation" => "man/installation.md",
            "Background" => "man/background.md",
            "Getting started" => "man/getting_started.md",
            "Evolutionary models" => "man/evolutionary_models.md",
            "Cluster graphs" => "man/clustergraphs.md",
            "Message passing" => "man/message_passing.md",
            "Message schedules" => "man/message_schedules.md"
        ]
    ]
)

# deploydocs(;
#     repo="github.com/cecileane/PhyloGaussianBeliefProp.jl",
#     devbranch="main",
# )
