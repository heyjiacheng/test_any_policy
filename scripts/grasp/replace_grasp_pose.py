import json
import numpy as np
import math

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def extract_p_R(matrix):
    """
    Extracts Translation vector (p) and Rotation matrix (R) 
    from a 4x4 homogeneous transformation matrix.
    """
    mat = np.array(matrix)
    p = mat[:3, 3]       # Position (x, y, z)
    R = mat[:3, :3]      # Rotation (3x3 matrix)
    return p, R

def geodesic_distance(R1, R2):
    """
    Calculates the geodesic rotation angle theta between two rotation matrices.
    theta(R1, R2) = arccos((trace(R1.T @ R2) - 1) / 2)
    """
    # Calculate relative rotation
    R_rel = np.dot(R1.T, R2)
    
    # Calculate trace
    trace_val = np.trace(R_rel)
    
    # Numerical stability: clip value to [-1, 1] to avoid NaN in arccos
    cos_theta = (trace_val - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    return np.arccos(cos_theta)

def calculate_cost(p_cand, R_cand, p_ref, R_ref, alpha_p=1.0, alpha_r=1.0):
    """
    Calculates the cost (the term inside the exp function of the formula).
    Minimizing this cost is equivalent to maximizing the score S.
    """
    # Euclidean distance squared for position
    dist_p_sq = np.sum((p_cand - p_ref)**2)
    
    # Geodesic angle squared for rotation
    theta = geodesic_distance(R_cand, R_ref)
    dist_r_sq = theta**2
    
    # Weighted sum (The exponent argument in the formula)
    cost = (alpha_p * dist_p_sq) + (alpha_r * dist_r_sq)
    return cost

def main():
    # 1. Load Data
    grasps_data = load_json('scripts/grasp/inputs/grasps_0_scene.json')
    trajectory_data = load_json('scripts/grasp/inputs/trajectory.json')

    # 2. Get Reference Grasp (g_N-1)
    # The prompt asks to compare against the last step of the trajectory
    last_step = trajectory_data[-1]
    ref_matrix = last_step['tcp_matrix']
    p_ref, R_ref = extract_p_R(ref_matrix)

    print(f"Reference Grasp (Step {last_step['step']}) loaded.")

    # 3. Iterate through Candidate Grasps (g_hat)
    candidates = grasps_data['grasps']

    best_grasp = None
    best_grasp_matrix_converted = None
    min_cost = float('inf')

    # Weights (assumed 1.0 as they are not specified in the prompt)
    # You can adjust alpha_p (position weight) and alpha_r (rotation weight)
    alpha_p = 1.0
    alpha_r = 1.0

    # Coordinate system transformation matrix
    # grasps: X=closing, Y=finger, Z=approach
    # trajectory: -X=main body (approach), ±Z=fingers (closing), Y=vertical
    # Transform: grasps_X -> traj_Z, grasps_Y -> traj_Y, grasps_Z -> traj_-X
    transform_matrix = np.array([
        [ 0,  0, -1,  0],
        [ 0,  1,  0,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ])

    # Local offset in grasps coordinate system: move 10cm (0.1m) along Z-axis
    local_offset = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.10],  # Z-axis offset: 10cm forward
        [0, 0, 0, 1]
    ])

    for grasp in candidates:
        cand_matrix = np.array(grasp['transform_matrix'])

        # Apply local offset in grasps coordinate system (right multiply)
        cand_matrix_offset = cand_matrix @ local_offset

        # Convert from grasps coordinate system to trajectory coordinate system
        # Right multiply to transform the local coordinate frame definition
        # Similar to: tcp_matrix_world @ o3d_to_sapien_rotation
        cand_matrix_converted = cand_matrix_offset @ transform_matrix

        p_cand, R_cand = extract_p_R(cand_matrix_converted)

        # Calculate the metric defined in the image
        cost = calculate_cost(p_cand, R_cand, p_ref, R_ref, alpha_p, alpha_r)

        if cost < min_cost:
            min_cost = cost
            best_grasp = grasp
            best_grasp_matrix_converted = cand_matrix_converted.tolist()

    if best_grasp:
        print(f"Best Match Found:")
        print(f"  Grasp ID: {best_grasp['grasp_id']}")
        print(f"  Original Score: {best_grasp['score']}")
        print(f"  Difference Cost: {min_cost:.6f}")

        # 4. Replace the matrix in trajectory
        # Update the last step's matrix with the converted best candidate's matrix
        trajectory_data[-1]['tcp_matrix'] = best_grasp_matrix_converted

        # 5. Save the result
        output_filename = 'scripts/grasp/outputs/trajectory_updated.json'
        save_json(trajectory_data, output_filename)
        print(f"Updated trajectory saved to {output_filename}")
    else:
        print("No suitable grasp found.")

if __name__ == "__main__":
    main()