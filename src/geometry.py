from openff.toolkit.topology import Molecule
import numpy as np

def extract_internal_indices(off_mol: Molecule):
    # Bond indices
    bond_idx = np.array(
        [(b.atom1.molecule_atom_index, b.atom2.molecule_atom_index)
         for b in off_mol.bonds],
        dtype=int
    )

    # Angle indices
    angle_idx = np.array(
        [(a[0].molecule_atom_index, a[1].molecule_atom_index, a[2].molecule_atom_index)
         for a in off_mol.angles],
        dtype=int
    )

    # Proper torsions
    proper_idx = np.array(
        [(t[0].molecule_atom_index, t[1].molecule_atom_index,
          t[2].molecule_atom_index, t[3].molecule_atom_index)
         for t in off_mol.propers],
        dtype=int
    )

    # Improper torsions
    improper_idx = np.array(
        [(t[0].molecule_atom_index, t[1].molecule_atom_index,
          t[2].molecule_atom_index, t[3].molecule_atom_index)
         for t in off_mol.impropers],
        dtype=int
    )

    return {
        "bonds": bond_idx,
        "angles": angle_idx,
        "propers": proper_idx,
        "impropers": improper_idx,
    }


def compute_bond_lengths(coords, bond_idx):
    # coords: (N,3); bond_idx: (M,2)
    v = coords[bond_idx[:, 1]] - coords[bond_idx[:, 0]]
    return np.linalg.norm(v, axis=1)   # same length units as coords


def compute_angles(coords, angle_idx):
    # Returns angles in radians
    i = angle_idx[:, 0]
    j = angle_idx[:, 1]
    k = angle_idx[:, 2]

    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]

    # normalize
    v1 /= np.linalg.norm(v1, axis=1)[:, None]
    v2 /= np.linalg.norm(v2, axis=1)[:, None]

    cos_theta = np.einsum("ij,ij->i", v1, v2)
    # clamp for numerical safety
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)        # radians


def compute_dihedrals(coords, torsion_idx):
    """
    Proper/improper torsion angles in radians.
    torsion_idx: (M,4) with (i,j,k,l)
    """
    i = torsion_idx[:, 0]
    j = torsion_idx[:, 1]
    k = torsion_idx[:, 2]
    l = torsion_idx[:, 3]

    p0 = coords[i]
    p1 = coords[j]
    p2 = coords[k]
    p3 = coords[l]

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 for stability
    b1_norm = np.linalg.norm(b1, axis=1)[:, None]
    b1_unit = b1 / b1_norm

    # perpendicular components
    v = b0 - np.einsum("ij,ij->i", b0, b1_unit)[:, None] * b1_unit
    w = b2 - np.einsum("ij,ij->i", b2, b1_unit)[:, None] * b1_unit

    v /= np.linalg.norm(v, axis=1)[:, None]
    w /= np.linalg.norm(w, axis=1)[:, None]

    x = np.einsum("ij,ij->i", v, w)
    y = np.einsum("ij,ij->i", np.cross(b1_unit, v), w)

    return np.arctan2(y, x)   # radians in (-π, π)

def compute_internal_coordinates(off_mol, coords):
    idx = extract_internal_indices(off_mol)

    bonds    = compute_bond_lengths(coords, idx["bonds"])
    angles   = compute_angles(coords, idx["angles"])
    propers  = compute_dihedrals(coords, idx["propers"])
    improps  = compute_dihedrals(coords, idx["impropers"])

    return {
        "bonds": bonds,        # length array
        "angles": angles,      # radians
        "propers": propers,    # radians
        "impropers": improps,  # radians
        "indices": idx,
    }
