# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # must be set before importing torch/espaloma

import gc
import multiprocessing as mp

import numpy    as np
import mdtraj   as md
import qcportal
import espaloma as esp
import matplotlib.pyplot as plt

from openmm                  import VerletIntegrator, LocalEnergyMinimizer, Platform
from openmm.app              import Simulation
from openmm.unit             import picosecond, kilojoules_per_mole, nanometer
from openff.toolkit.topology import Molecule, Topology

from src.geometry import compute_internal_coordinates
from src.stats    import freedman_diaconis_bins, histogram_cdf

import torch

#%%
# ---- Global (per-process) lazy objects ----
_client   = None
_dataset  = None
_esp_model = None

QCARCHIVE_URL = "https://api.qcarchive.molssi.org:443"
DATASET_NAME  = "OpenFF Industry Benchmark Season 1 v1.2"
DATASET_TYPE  = "optimization"

ang_to_nm = 0.1

#%%
n_logical = 8#os.cpu_count()
THREADS_PER_SIM = 2
N_PROCS         = max(1, n_logical // THREADS_PER_SIM)
BATCH_SIZE      = int(N_PROCS * 2) 

# ---- Utility functions (unchanged logic) ----

def compute_rmsd(coords_A, coords_B):
    traj0 = md.Trajectory(coords_A[None, :, :], None)
    traj1 = md.Trajectory(coords_B[None, :, :], None)
    return md.rmsd(traj1, traj0)[0]


def rmsd_1d(x0, x1):
    return np.sqrt(np.mean((x1 - x0) ** 2))


def rmsd_periodic(phi0, phi1):
    d = phi1 - phi0
    d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
    return np.sqrt(np.mean(d ** 2))


# ---- Per-process lazy initialisation ----

def get_client():
    global _client
    if _client is None:
        _client = qcportal.PortalClient(QCARCHIVE_URL)
    return _client


def get_dataset():
    global _dataset
    if _dataset is None:
        client = get_client()
        _dataset = client.get_dataset(
            dataset_name=DATASET_NAME,
            dataset_type=DATASET_TYPE,
        )
    return _dataset


def get_esp_model():
    global _esp_model
    if _esp_model is None:
        model = esp.get_model("latest")
        model.to("cpu")
        _esp_model = model
    return _esp_model


# ---- Core single-molecule computation (logic from your loop) ----
def make_simulation(topology, system):
    integrator = VerletIntegrator(0.002 * picosecond)
    platform   = Platform.getPlatformByName("CPU")

    properties = {"Threads": str(THREADS_PER_SIM)}  # key name is "Threads" for CPU platform 

    sim = Simulation(topology, system, integrator, platform, properties)
    return sim

def process_single_molecule(name: str):
    dataset   = get_dataset()
    esp_model = get_esp_model()

    entry  = dataset.get_entry(entry_name=name)
    coords = entry.dict()["initial_molecule"]["geometry"] * 0.1  # nm
    mol    = Molecule.from_qcschema(entry)
    top    = Topology.from_molecules(molecules=[mol])

    mgraph = esp.Graph(mol)
    mgraph.heterograph = mgraph.heterograph.to("cpu")

    with torch.no_grad():
        esp_model(mgraph.heterograph)

    system = esp.graphs.deploy.openmm_system_from_graph(mgraph)
    simulation = make_simulation(top, system)

    simulation.context.setPositions(coords)

    # Initial state
    state  = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
    energy = state.getPotentialEnergy()
    forces = state.getForces(asNumpy=True)
    forces = forces.value_in_unit(kilojoules_per_mole / nanometer).astype(np.float32)
    coords0 = state.getPositions(asNumpy=True)
    coords0 = coords0.value_in_unit(nanometer).astype(np.float32)
    ic0     = compute_internal_coordinates(mol, coords0)

    # Minimize
    LocalEnergyMinimizer.minimize(
        simulation.context,
        tolerance=10 * kilojoules_per_mole / nanometer,
        maxIterations=10_000,
    )

    # Minimized state
    state_minim = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
    energy_min  = state_minim.getPotentialEnergy()
    forces_min  = state_minim.getForces(asNumpy=True)
    forces_min  = forces_min.value_in_unit(kilojoules_per_mole / nanometer).astype(np.float32)
    coords1     = state_minim.getPositions(asNumpy=True)
    coords1     = coords1.value_in_unit(nanometer).astype(np.float32)
    ic1         = compute_internal_coordinates(mol, coords1)

    # Cartesian RMSD
    rmsd_cart = compute_rmsd(coords1, coords0)

    # Internal-coordinate RMSDs
    rmsd_bonds     = rmsd_1d(ic1["bonds"],     ic0["bonds"])
    rmsd_angles    = rmsd_periodic(ic1["angles"],    ic0["angles"])
    rmsd_propers   = rmsd_periodic(ic1["propers"],   ic0["propers"])
    rmsd_impropers = rmsd_periodic(ic1["impropers"], ic0["impropers"])

    # Clean up local heavy objects
    del state, state_minim, forces, forces_min, coords0, coords1, ic0, ic1
    del system, simulation, mgraph
    gc.collect()

    # Return whatever you wish to keep
    return {
        "name": name,
        "rmsd_cart": rmsd_cart,
        "rmsd_bonds": rmsd_bonds,
        "rmsd_angles": rmsd_angles,
        "rmsd_propers": rmsd_propers,
        "rmsd_impropers": rmsd_impropers,
        "energy_initial": energy,
        "energy_min": energy_min,
    }


# ---- Batch worker for multiprocessing ----

def process_batch(name_batch):
    batch_results = []
    for name in name_batch:
        try:
            res = process_single_molecule(name)
            batch_results.append(res)
        except Exception as e:
            # You can log or store the error here if desired
            batch_results.append({
                "name": name,
                "error": repr(e),
            })
    return batch_results


# ---- Main parallel driver ----

def chunk_list(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():

    os.environ["OPENMM_CPU_THREADS"] = str(THREADS_PER_SIM)

    print(f"Using {N_PROCS} processes × {THREADS_PER_SIM} OpenMM threads = "
          f"{N_PROCS * THREADS_PER_SIM} total threads")
    
    # Get names once in the parent process
    # build list of molecule names
    client  = qcportal.PortalClient(QCARCHIVE_URL)
    dataset = client.get_dataset(dataset_name=DATASET_NAME, dataset_type=DATASET_TYPE)
    names   = list(dataset.entry_names)

    batches = list(chunk_list(names, BATCH_SIZE))

    with mp.Pool(processes=N_PROCS, maxtasksperchild=10) as pool:
        all_results = []
        i = 1
        for batch_res in pool.map(process_batch, batches):
            print(i)
            all_results.extend(batch_res)
            i+=1

    filtered = [r for r in all_results if "error" not in r]

    rmsd_array = np.array([
        [
            r["rmsd_cart"],
            r["rmsd_bonds"],
            r["rmsd_angles"],
            r["rmsd_propers"],
            r["rmsd_impropers"],
        ]
        for r in filtered
    ])

    # You can save this, plot histograms, etc.
    # np.save("espaloma_rmsds.npy", rmsd_array)
    return rmsd_array

#%%
if __name__ == "__main__":
    rmsd_array = main()


# %%
def graph_rmsd(rmsd,
               title    = "Structure",
               xlabel   = "RMSD / nm",
               ylabel   = "P(RMSD)",
               ylabel_2 = "CDF(RMSD)"):

    fig, ax = plt.subplots(figsize = (7,7), layout = "tight")
    ax_2 = ax.twinx()

    ax.set_title(title, fontsize = 20, fontweight = "bold")

    n, b, _ = ax.hist(rmsd, bins = freedman_diaconis_bins(rmsd),
            density=True, color = "royalblue", edgecolor="k")

    cdf = histogram_cdf(n, b)
    ax_2.plot(b, cdf, c ="k")
    ax_2.set_ylim(0)

    ax.set_xlabel(xlabel, fontsize = 18, fontweight = "bold")
    ax.set_ylabel(ylabel, fontsize = 18, fontweight = "bold")
    ax_2.set_ylabel(ylabel_2, fontsize = 18, fontweight = "bold")
    ax.tick_params(labelsize = 16)
    ax_2.tick_params(labelsize = 16)

#%%
graph_rmsd(rmsd_array[:,0], title= "Structure", xlabel = "RMSD / nm")
graph_rmsd(rmsd_array[:,1], title= "Bonds",     xlabel = "RMSD / nm")
graph_rmsd(rmsd_array[:,2], title= "Angles",    xlabel = "RMSD / rad")
graph_rmsd(rmsd_array[:,3], title= "Propers",   xlabel = "RMSD / rad")
graph_rmsd(rmsd_array[:,4], title= "Impropers", xlabel = "RMSD / rad")

# %%
