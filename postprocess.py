#%%
import matplotlib.pyplot as plt

from itertools import combinations
from src.io    import BenchmarkResults
from src.stats import freedman_diaconis_bins, histogram_cdf

# %%
openff_data   = "openff_2048.hdf5"
espaloma_data = "espaloma_2048.hdf5"

openff_bmark   = BenchmarkResults.from_hdf5(openff_data)
espaloma_bmark = BenchmarkResults.from_hdf5(espaloma_data)
# %%
def get_conformer_groups(dataset):
    groups  = []
    current = dataset.smiles[0]
    buff    = [0]
    for i, smiles in enumerate(dataset.smiles[1:], start = 1):
        if smiles != current:
            current = smiles
            groups.append(buff)
            buff = [i]
            continue
        buff.append(i)
    return groups

# %%
conformers_openff   = get_conformer_groups(openff_bmark)
conformers_espaloma = get_conformer_groups(espaloma_bmark)

pairs_openff   = [list(combinations(_, 2)) for _ in conformers_openff]
pairs_espaloma = [list(combinations(_, 2)) for _ in conformers_espaloma]

# %%
dE_openff = []
for pairs in pairs_openff:
    for i,j in pairs:
        emin_i = openff_bmark.energy_min[i]
        emin_j = openff_bmark.energy_min[j]
        demin  = emin_i - emin_j

        eqm_i = openff_bmark.energy_initial[i]
        eqm_j = openff_bmark.energy_initial[j]
        deqm  = eqm_i - eqm_j

        dde = deqm - demin
        dE_openff.append(dde)

dE_espaloma = []
for pairs in pairs_espaloma:
    for i,j in pairs:
        emin_i = espaloma_bmark.energy_min[i]
        emin_j = espaloma_bmark.energy_min[j]
        demin  = emin_i - emin_j

        eqm_i = espaloma_bmark.energy_initial[i]
        eqm_j = espaloma_bmark.energy_initial[j]
        deqm  = eqm_i - eqm_j

        dde = deqm - demin
        dE_espaloma.append(dde)


# %%

fig, ax = plt.subplots(figsize = (7,7), layout = "tight")
ax2 = ax.twinx()

n, b, _ = ax.hist(dE_openff, bins = freedman_diaconis_bins(dE_openff), density= True, color = "royalblue", edgecolor = "k", alpha = 0.5)
cdf = histogram_cdf(n, b)
ax2.plot(b, cdf, c = "royalblue", lw = 3)

n, b, _ = ax.hist(dE_espaloma, bins = freedman_diaconis_bins(dE_espaloma), density= True, color = "firebrick", edgecolor = "k", alpha = 0.5)
cdf = histogram_cdf(n, b)
ax2.plot(b, cdf, c = "firebrick", lw = 3)

ax2.set_ylim(0)

# %%
