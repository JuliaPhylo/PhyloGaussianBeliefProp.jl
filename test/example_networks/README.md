# Example networks
The `Source` column indicates which figure of which study the network was coded from.

| Network | Source |
| --- | --- |
| `lazaridis_2014.phy` | [Fig 3](https://doi.org/10.1038/nature13673) |
| `lipson_2020b.phy` | [Extended Data Fig 4](https://doi.org/10.1038/s41586-020-1929-1) |
| `mateescu_2010.phy` | [Fig 2a](https://doi.org/10.1613/jair.2842) |
| `muller_2022.phy` | [Fig 1a](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9297283/) |
| `sun_2023.phy` | [Fig 4c](https://doi.org/10.1038/s41559-023-02185-8) |
| `maier_2023.phy` | [Fig 1e](https://doi.org/10.7554/eLife.85492)|

Networks (coded in extended newick format) were preprocessed as follows:
- `mateescu_2020.phy`, `lazaridis_2014.phy`: edge lengths and inheritance
probabilities were set arbitrarily.
- `muller_2022.phy`: the inheritance probabilities were estimated from the
inferred recombination breakpoints (see [muller2022_nexus2newick.jl](https://github.com/bstkj/graphicalmodels_for_phylogenetics_code/blob/5f61755c4defe804fd813113e883d49445971ade/real_networks/muller2022_nexus2newick.jl)).
- `lipson_2020b.phy`: degree-2 nodes were suppressed and any resulting
length 0 edges assigned length 1 (see [lipson2020b_notes.jl](https://github.com/bstkj/graphicalmodels_for_phylogenetics_code/blob/5f61755c4defe804fd813113e883d49445971ade/real_networks/lipson2020b_notes.jl)).
- `sun_2023.phy`: degree-2 nodes were not suppressed. All admixture edges
were assigned a length of 1.0 (the minimum length among all drift edges).
- `maier_2023.phy`: Edge lengths were set arbitrarily.