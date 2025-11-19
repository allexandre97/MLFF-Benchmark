import h5py
import numpy as np

class BenchmarkResults:
    """
    Compact container for benchmark results, with efficient HDF5 I/O.

    Coordinates are stored in ragged form:
        coords_qm  : (total_atoms, 3)
        coords_min : (total_atoms, 3)
        n_atoms    : (n_mols,)
        offsets    : (n_mols+1,)
    """
    def __init__(self, backend: str, dataset_name: str):
        self.backend = backend
        self.dataset_name = dataset_name

        self.names   = []
        self.smiles  = []
        self.n_atoms = []

        self._coords_qm_chunks  = []
        self._coords_min_chunks = []

        self.rmsd_cart      = []
        self.rmsd_bonds     = []
        self.rmsd_angles    = []
        self.rmsd_propers   = []
        self.rmsd_impropers = []
        self.tfd            = []

        self.energy_initial = []
        self.energy_min     = []

    def add(self, record: dict):
        """
        Add one molecule result as returned by process_single_molecule.
        """
        name   = record["name"]
        smiles = record["smiles"]

        coords_qm  = record["coords"]["qm"]
        coords_min = record["coords"]["min"]

        n = coords_qm.shape[0]
        assert coords_qm.shape == coords_min.shape == (n, 3)

        self.names.append(name)
        self.smiles.append(smiles)
        self.n_atoms.append(n)

        self._coords_qm_chunks.append(coords_qm.astype(np.float32, copy=False))
        self._coords_min_chunks.append(coords_min.astype(np.float32, copy=False))

        self.rmsd_cart.append(float(record["rmsd_cart"]))
        self.rmsd_bonds.append(float(record["rmsd_bonds"]))
        self.rmsd_angles.append(float(record["rmsd_angles"]))
        self.rmsd_propers.append(float(record["rmsd_propers"]))
        self.rmsd_impropers.append(float(record["rmsd_impropers"]))
        self.tfd.append(float(record["tfd"]))

        self.energy_initial.append(float(record["energy_initial"]))
        self.energy_min.append(float(record["energy_min"]))

    def _finalize_coords(self):
        """
        Build concatenated coordinate arrays and offsets.
        """
        if not self._coords_qm_chunks:
            self.coords_qm  = np.zeros((0, 3), dtype=np.float32)
            self.coords_min = np.zeros((0, 3), dtype=np.float32)
            self.offsets    = np.zeros(1, dtype=np.int64)
            self.n_atoms    = np.zeros(0, dtype=np.int64)
            return

        self.n_atoms = np.asarray(self.n_atoms, dtype=np.int64)
        self.coords_qm  = np.vstack(self._coords_qm_chunks)
        self.coords_min = np.vstack(self._coords_min_chunks)

        self.offsets = np.zeros(len(self.n_atoms) + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.n_atoms)

    def to_hdf5(self, filename: str, compression: str = "gzip"):
        """
        Save all results to an HDF5 file for later postprocessing.
        """
        self._finalize_coords()

        names  = np.asarray(self.names,  dtype=h5py.string_dtype(encoding="utf-8"))
        smiles = np.asarray(self.smiles, dtype=h5py.string_dtype(encoding="utf-8"))

        rmsd_cart      = np.asarray(self.rmsd_cart,      dtype=np.float32)
        rmsd_bonds     = np.asarray(self.rmsd_bonds,     dtype=np.float32)
        rmsd_angles    = np.asarray(self.rmsd_angles,    dtype=np.float32)
        rmsd_propers   = np.asarray(self.rmsd_propers,   dtype=np.float32)
        rmsd_impropers = np.asarray(self.rmsd_impropers, dtype=np.float32)
        tfd            = np.asarray(self.tfd,            dtype=np.float32)

        energy_initial = np.asarray(self.energy_initial, dtype=np.float32)
        energy_min     = np.asarray(self.energy_min,     dtype=np.float32)

        with h5py.File(filename, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["backend"]      = self.backend
            meta.attrs["dataset_name"] = self.dataset_name

            f.create_dataset("names",  data=names)
            f.create_dataset("smiles", data=smiles)

            f.create_dataset("n_atoms", data=self.n_atoms)
            f.create_dataset("offsets", data=self.offsets)

            f.create_dataset("coords_qm",  data=self.coords_qm,
                             compression=compression)
            f.create_dataset("coords_min", data=self.coords_min,
                             compression=compression)

            f.create_dataset("rmsd_cart",      data=rmsd_cart)
            f.create_dataset("rmsd_bonds",     data=rmsd_bonds)
            f.create_dataset("rmsd_angles",    data=rmsd_angles)
            f.create_dataset("rmsd_propers",   data=rmsd_propers)
            f.create_dataset("rmsd_impropers", data=rmsd_impropers)
            f.create_dataset("tfd",            data=tfd)

            f.create_dataset("energy_initial", data=energy_initial)
            f.create_dataset("energy_min",     data=energy_min)

    @classmethod
    def from_hdf5(cls, filename: str) -> "BenchmarkResults":
        """
        Load results from an HDF5 file created by to_hdf5.
        """
        with h5py.File(filename, "r") as f:
            meta = f["meta"]
            backend      = meta.attrs["backend"]
            dataset_name = meta.attrs["dataset_name"]

            obj = cls(backend=backend, dataset_name=dataset_name)

            obj.names   = [s for s in f["names"][()]]
            obj.smiles  = [s for s in f["smiles"][()]]
            obj.n_atoms = f["n_atoms"][()]
            obj.offsets = f["offsets"][()]

            obj.coords_qm  = f["coords_qm"][()]
            obj.coords_min = f["coords_min"][()]

            obj.rmsd_cart      = f["rmsd_cart"][()].tolist()
            obj.rmsd_bonds     = f["rmsd_bonds"][()].tolist()
            obj.rmsd_angles    = f["rmsd_angles"][()].tolist()
            obj.rmsd_propers   = f["rmsd_propers"][()].tolist()
            obj.rmsd_impropers = f["rmsd_impropers"][()].tolist()
            obj.tfd            = f["tfd"][()].tolist()

            obj.energy_initial = f["energy_initial"][()].tolist()
            obj.energy_min     = f["energy_min"][()].tolist()

        return obj

    def get_coords_qm_for_mol(self, idx: int) -> np.ndarray:
        """
        Convenience accessor: QM coordinates for molecule `idx`.
        """
        start = self.offsets[idx]
        end   = self.offsets[idx + 1]
        return self.coords_qm[start:end]

    def get_coords_min_for_mol(self, idx: int) -> np.ndarray:
        """
        Convenience accessor: minimized coordinates for molecule `idx`.
        """
        start = self.offsets[idx]
        end   = self.offsets[idx + 1]
        return self.coords_min[start:end]
