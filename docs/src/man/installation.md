# Installation

For information on how to install Julia and PhyloNetworks, see
[here](https://crsl4.github.io/PhyloNetworks.jl/dev/man/installation/#Installation).
PhyloGaussianBeliefProp depends on PhyloNetworks.

To install [PhyloGaussianBeliefProp](https://github.com/JuliaPhylo/PhyloGaussianBeliefProp.jl)
in the Julia REPL, do:
```julia
julia> using Pkg

julia> Pkg.add("PhyloGaussianBeliefProp")
```

Or enter `]` in the Julia REPL to access the package mode, and do:
```
pkg> add PhyloGaussianBeliefProp
```

In this manual, we will also use
[PhyloNetworks](https://github.com/crsl4/PhyloNetworks.jl) and other packages,
to be installed similarly, here in package mode:
```
pkg> add PhyloNetworks

pkg> add DataFrames
```
