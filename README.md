OTMalign: A Fast and Robust TM-score-driven Protein Structure Alignment Algorithm using Optimal Transport
=========================================================================================================

Introduction
------------

OTMalign is a novel algorithm for protein structure alignment that iteratively optimizes the TM-score. It leverages the theory of Optimal Transport (OT) to establish a "soft" correspondence between protein structures, guided by a TM-score-inspired reward matrix. This approach avoids the pitfalls of combinatorial searches and provides a global perspective at each iteration, leading to a fast and robust alignment.

The core of the algorithm is an iterative process:
1.  **Soft Correspondence:** At each step, the Sinkhorn algorithm is used to find an optimal transport plan (the P matrix). This matrix represents a soft, probabilistic matching between the residues of the two proteins, based on a TM-score-like reward function.
2.  **Transformation Update:** The optimal rigid body transformation (rotation and translation) is then determined analytically using a weighted Kabsch algorithm. The weights are derived from the P matrix, meaning the transformation is guided by the most likely residue correspondences.

This iterative process of finding a global soft correspondence and then analytically solving for the transformation is computationally efficient and robustly converges to a high-quality alignment.

Usage
-----

`python OTMalign.py --mobile <mobile.pdb> --reference <reference.pdb> --output <aligned.pdb> [OPTIONS]`

Parameters
----------

**File I/O Arguments:**

*   `--mobile`: Path to the mobile PDB file. (Required)
*   `--reference`: Path to the reference PDB file. (Required)
*   `--output`: Path to save the aligned mobile PDB file. (Default: `aligned_mobile.pdb`)
*   `--mobile_chain`: Chain ID for the mobile protein (optional).
*   `--reference_chain`: Chain ID for the reference protein (optional).

**Algorithm Hyperparameters:**

*   `--n_iters`: Maximum number of ICP iterations. (Default: 100)
*   `--tm_conv_tol`: Convergence tolerance for the change in TM-score. (Default: 1e-5)
*   `--gamma`: Sharpness factor for the Sinkhorn algorithm. (Default: 20.0)
*   `--sinkhorn_iters`: Number of inner iterations for the Sinkhorn algorithm. (Default: 10)
*   `--cutoff`: Distance cutoff for the reward matrix sigmoid weight. (Default: 7.0)
*   `--steepness`: Steepness of the sigmoid cutoff. (Default: 2.0)

Example Command from the Paper
--------------------------------

The following command was used to generate the results for the 40 protein pairs from the RPIC database, as presented in the paper:

`python OTMalign.py --mobile <path_to_mobile.pdb> --reference <path_to_reference.pdb> --output <path_to_aligned.pdb> --n_iters 100 --tm_conv_tol 1e-5 --gamma 20.0 --sinkhorn_iters 10 --cutoff 7.0 --steepness 2.0`
