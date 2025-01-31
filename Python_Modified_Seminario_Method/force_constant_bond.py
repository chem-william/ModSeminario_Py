import numpy as np


def force_constant_bond(atom_A, atom_B, eigenvalues, eigenvectors, coords):
    """
    Force Constant - Equation 10 of Seminario paper - gives force
    constant for bond
    """
    eigenvalues_AB = eigenvalues[atom_A, atom_B, :]
    eigenvectors_AB = eigenvectors[:, :, atom_A, atom_B]

    # Vector along bond
    diff_AB = coords[atom_B, :] - coords[atom_A, :]
    norm_diff_AB = np.linalg.norm(diff_AB)
    unit_vectors_AB = diff_AB / norm_diff_AB

    # Projections of eigenvalues
    k_AB = 0
    for i in range(0, 3):
        dot_product = np.abs(np.dot(unit_vectors_AB, eigenvectors_AB[:, i]))
        k_AB += eigenvalues_AB[i] * dot_product

    k_AB = -k_AB * 0.5  # Convert to OPLS form

    return k_AB