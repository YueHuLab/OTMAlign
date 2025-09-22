import torch
import numpy as np
import argparse

# --- PDB Parsing (from reference) ---
def parse_pdb_ca(file_path):
    coords, atom_lines = [], []
    seen_residues = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                    continue
                
                alt_loc = line[16]
                if alt_loc not in (' ', 'A'):
                    continue

                residue_uid = (line[21], line[22:27])
                if residue_uid in seen_residues:
                    continue
                seen_residues.add(residue_uid)
                
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                atom_lines.append(line)
    except FileNotFoundError:
        print(f"Error: PDB file not found at {file_path}")
        return None, None
    if not coords:
        print(f"Error: No C-alpha atoms found in {file_path}.")
        return None, None
    return torch.tensor(coords, dtype=torch.float32), atom_lines

def write_pdb(file_path, atom_lines, coords):
    with open(file_path, 'w') as f:
        for i, line in enumerate(atom_lines):
            if i < len(coords):
                x, y, z = coords[i]
                f.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")

# --- Core Algorithm (combining reference and ICP) ---

def get_d0(length):
    return 1.24 * (length - 15)**(1/3) - 1.8 if length > 15 else 0.5

def sinkhorn_iterations(M, num_iters=10):
    P = M
    for _ in range(num_iters):
        P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
        P = P / (P.sum(dim=0, keepdim=True) + 1e-8)
    return P

def compute_weighted_kabsch(P_matrix, mobile_coords, ref_coords):
    # P_matrix is (N, M), mobile is (N, 3), ref is (M, 3)
    
    # Transpose P to get weights for each point cloud
    P_T = P_matrix.T # (M, N) 
    
    # Calculate weighted centroids
    centroid_mob = torch.sum(mobile_coords * P_matrix.sum(dim=1, keepdim=True), dim=0) / P_matrix.sum()
    centroid_ref = torch.sum(ref_coords * P_T.sum(dim=1, keepdim=True), dim=0) / P_T.sum()

    # Center the coordinates
    mobile_centered = mobile_coords - centroid_mob
    ref_centered = ref_coords - centroid_ref

    # Compute covariance matrix
    H = mobile_centered.T @ P_matrix @ ref_centered

    # SVD
    U, _, Vt = torch.svd(H)
    R = Vt @ U.T

    # Handle reflection case
    if torch.det(R) < 0:
        Vt[:, -1] *= -1
        R = Vt @ U.T

    t = centroid_ref - R @ centroid_mob
    return R, t

def parse_pdb(file_path, chain_id=None, c_alpha_only=True):
    """A robust PDB parser that handles alternative locations, chain IDs, and full-atom parsing."""
    coords, atom_lines = [], []
    seen_residues = set() # Only used for C-alpha only mode
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                
                if c_alpha_only and line[12:16].strip() != "CA":
                    continue

                if chain_id and line[21].strip() != chain_id:
                    continue

                alt_loc = line[16]
                if alt_loc not in (' ', 'A'):
                    continue

                if c_alpha_only:
                    residue_uid = (line[21], line[22:27])
                    if residue_uid in seen_residues:
                        continue
                    seen_residues.add(residue_uid)
                
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                atom_lines.append(line)
    except FileNotFoundError:
        print(f"Error: PDB file not found at {file_path}")
        return None, None
    if not coords:
        print(f"Error: No matching atoms found in {file_path} for the specified criteria.")
        return None, None
    return torch.tensor(coords, dtype=torch.float32), atom_lines

def decode_alignment_matrix(P):
    """Extracts the most likely one-to-one correspondences from the soft assignment matrix P."""
    aligned_pairs = []
    used_cols = set()
    row_indices = torch.arange(P.shape[0])
    best_cols = torch.argmax(P, dim=1)
    confidences = P[row_indices, best_cols]
    sorted_indices = torch.argsort(confidences, descending=True)
    for i in sorted_indices:
        j = best_cols[i]
        if j.item() not in used_cols:
            aligned_pairs.append((i.item(), j.item()))
            used_cols.add(j.item())
    return sorted(aligned_pairs)

def calculate_rmsd(coords1, coords2):
    return torch.sqrt(torch.mean(torch.sum((coords1 - coords2)**2, dim=1))).item()

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A TM-score driven, iterative protein structure alignment tool using Sinkhorn-ICP (v4: Full Atom Output).",
        formatter_class=argparse.RawTextHelpFormatter)
    
    # --- File I/O Arguments ---
    parser.add_argument("--mobile", required=True, help="Path to the mobile PDB file.")
    parser.add_argument("--reference", required=True, help="Path to the reference PDB file.")
    parser.add_argument("--output", default="aligned_mobile.pdb", help="Path to save the aligned mobile PDB file.")
    parser.add_argument("--mobile_chain", default=None, help="Chain ID for the mobile protein (optional).")
    parser.add_argument("--reference_chain", default=None, help="Chain ID for the reference protein (optional).")

    # --- Algorithm Hyperparameters ---
    parser.add_argument("--n_iters", type=int, default=100, help="Maximum number of ICP iterations.")
    parser.add_argument("--tm_conv_tol", type=float, default=1e-5, help="Convergence tolerance for the change in TM-score.")
    parser.add_argument("--gamma", type=float, default=20.0, help="Sharpness factor for the Sinkhorn algorithm.")
    parser.add_argument("--sinkhorn_iters", type=int, default=10, help="Number of inner iterations for the Sinkhorn algorithm.")
    parser.add_argument("--cutoff", type=float, default=7.0, help="Distance cutoff for the reward matrix sigmoid weight.")
    parser.add_argument("--steepness", type=float, default=2.0, help="Steepness of the sigmoid cutoff.")

    args = parser.parse_args()

    # --- Load and Prepare C-alpha Data for Alignment ---
    mobile_coords_ca, _ = parse_pdb(args.mobile, args.mobile_chain, c_alpha_only=True)
    ref_coords_ca, _ = parse_pdb(args.reference, args.reference_chain, c_alpha_only=True)

    if mobile_coords_ca is None or ref_coords_ca is None:
        exit()

    len_mobile, len_ref = len(mobile_coords_ca), len(ref_coords_ca)
    print(f"Aligning C-alpha trace of {args.mobile} (N={len_mobile}) to {args.reference} (M={len_ref})")

    # --- Sinkhorn-ICP Loop ---
    current_mobile_coords_ca = mobile_coords_ca.clone()
    final_P = None
    previous_tm_score = 0.0
    final_R, final_t = torch.eye(3), torch.zeros(3)
    
    print("\nStarting Sinkhorn-ICP iterations...")
    for i in range(args.n_iters):
        # 1. Calculate reward matrix
        d0 = get_d0(len_ref)
        dist_matrix = torch.cdist(current_mobile_coords_ca, ref_coords_ca)
        s_ij = 1.0 / (1.0 + (dist_matrix / d0)**2)
        w_ij = torch.sigmoid(-(dist_matrix - args.cutoff) * args.steepness)
        reward_matrix = s_ij * w_ij

        # 2. Get soft assignment matrix P from Sinkhorn
        P = sinkhorn_iterations(torch.exp(args.gamma * reward_matrix), num_iters=args.sinkhorn_iters)
        final_P = P

        # 3. Compute transformation using weighted Kabsch
        R, t = compute_weighted_kabsch(P, mobile_coords_ca, ref_coords_ca)
        final_R, final_t = R, t # Store the last transformation

        # 4. Apply transformation
        current_mobile_coords_ca = (R @ mobile_coords_ca.T).T + t

        # 5. Check for convergence based on TM-score change
        current_tm_score = (torch.sum(P * reward_matrix) / len_ref).item()
        score_change = current_tm_score - previous_tm_score
        
        print(f"Iteration {i+1:03d}: Soft TM-score = {current_tm_score:.6f}, Change = {score_change:.6f}")

        if i > 0 and score_change < args.tm_conv_tol:
            print(f"\nConverged after {i+1} iterations.")
            break
        previous_tm_score = current_tm_score

    # --- Final Analysis ---
    print("\n--- Final Analysis ---")
    aligned_pairs = decode_alignment_matrix(final_P)
    aligned_len = len(aligned_pairs)
    
    if aligned_len > 0:
        mobile_indices, ref_indices = zip(*aligned_pairs)
        mobile_indices, ref_indices = list(mobile_indices), list(ref_indices)
        
        # Final RMSD on aligned pairs
        final_rmsd = calculate_rmsd(current_mobile_coords_ca[mobile_indices], ref_coords_ca[ref_indices])
        print(f"Final C-alpha RMSD on {aligned_len} aligned pairs: {final_rmsd:.4f} Å")

        # Final TM-score on aligned pairs (normalized by both lengths)
        d_sq = torch.sum((current_mobile_coords_ca[mobile_indices] - ref_coords_ca[ref_indices])**2, dim=1)
        
        d0_ref = get_d0(len_ref)
        tm_score_ref = (1 / len_ref) * torch.sum(1.0 / (1.0 + d_sq / d0_ref**2))
        print(f"Final TM-score (normalized by reference, L={len_ref}): {tm_score_ref.item():.6f}")

        d0_mob = get_d0(len_mobile)
        tm_score_mob = (1 / len_mobile) * torch.sum(1.0 / (1.0 + d_sq / d0_mob**2))
        print(f"Final TM-score (normalized by mobile,   L={len_mobile}): {tm_score_mob.item():.6f}")

    else:
        print("\nCould not determine a final alignment.")

    # --- Write Full-Atom Output ---
    print(f"\nApplying transformation to full-atom model and saving to {args.output}...")
    mobile_coords_full, mobile_lines_full = parse_pdb(args.mobile, args.mobile_chain, c_alpha_only=False)
    if mobile_coords_full is not None:
        # Apply the final transformation to the full-atom coordinates
        transformed_full_coords = (final_R @ mobile_coords_full.T).T + final_t
        write_pdb(args.output, mobile_lines_full, transformed_full_coords)
        print("Success!")
    else:
        print("Could not parse full-atom coordinates. Skipping full-atom output.")