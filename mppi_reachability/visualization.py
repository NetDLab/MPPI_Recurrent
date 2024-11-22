import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_reachable_set(reachable_set, num_d=20, num_theta=20):
    if len(reachable_set) == 0:
        print("No reachable states to plot.")
        return

    # Convert reachable_set to NumPy array for efficient processing
    reachable_set_np = np.array(reachable_set)  # Shape: (num_reachable, 2)

    # Define the threshold for minimum square size based on grid step
    d_values = np.linspace(0.5, 5.0, num_d)
    theta_values = np.linspace(-np.pi, np.pi, num_theta)
    d_step = d_values[1] - d_values[0]
    theta_step = theta_values[1] - theta_values[0]

    threshold_d = d_step / 2  # Control minimum d direction length
    threshold_theta = theta_step / 2  # Control minimum theta direction length

    print(f"d_step: {d_step}, theta_step: {theta_step}")
    print(f"threshold_d: {threshold_d}, threshold_theta: {threshold_theta}")

    # Initialize list to hold final squares
    final_squares = []

    # Recursive subdivision function
    def subdivide_square(d_min, d_max, theta_min, theta_max, reachable_set_np, threshold_d, threshold_theta, squares):
        # Compute center point
        d_center = (d_min + d_max) / 2
        theta_center = (theta_min + theta_max) / 2

        # Tolerances
        tol_d = threshold_d
        tol_theta = threshold_theta

        # Check BRT status of center point
        center_in_brt = np.any((np.abs(reachable_set_np[:, 0] - d_center) < tol_d) &
                               (np.abs(reachable_set_np[:, 1] - theta_center) < tol_theta))

        # Define offsets to find adjacent center points
        offsets = [(-d_step, 0), (d_step, 0), (0, -theta_step), (0, theta_step),
                   (-d_step, -theta_step), (-d_step, theta_step), (d_step, -theta_step), (d_step, theta_step)]

        adjacent_in_brt = []
        for d_offset, theta_offset in offsets:
            adj_d_center = d_center + d_offset
            adj_theta_center = theta_center + theta_offset

            # Ensure adjacent center points are within bounds
            if not (0.5 <= adj_d_center <= 5.0) or not (-np.pi <= adj_theta_center <= np.pi):
                continue  # Skip out-of-bounds points

            adj_in_brt = np.any((np.abs(reachable_set_np[:, 0] - adj_d_center) < tol_d) &
                                (np.abs(reachable_set_np[:, 1] - adj_theta_center) < tol_theta))
            adjacent_in_brt.append(adj_in_brt)

        # Determine if subdivision is needed
        need_subdivide = False
        if center_in_brt:
            if not all(adjacent_in_brt):
                need_subdivide = True
        else:
            if any(adjacent_in_brt):
                need_subdivide = True

        # Check if square size is above threshold
        d_size = d_max - d_min
        theta_size = theta_max - theta_min

        if need_subdivide and d_size > threshold_d and theta_size > threshold_theta:
            # Subdivide into 4 smaller squares (2x2 grid for efficiency)
            d_mid = (d_min + d_max) / 2
            theta_mid = (theta_min + theta_max) / 2

            subdivide_square(d_min, d_mid, theta_min, theta_mid,
                             reachable_set_np, threshold_d, threshold_theta, squares)
            subdivide_square(d_min, d_mid, theta_mid, theta_max,
                             reachable_set_np, threshold_d, threshold_theta, squares)
            subdivide_square(d_mid, d_max, theta_min, theta_mid,
                             reachable_set_np, threshold_d, threshold_theta, squares)
            subdivide_square(d_mid, d_max, theta_mid, theta_max,
                             reachable_set_np, threshold_d, threshold_theta, squares)
        else:
            # Assign BRT status based on center point
            squares.append((d_min, d_max, theta_min, theta_max, center_in_brt))

    # Start subdivision with the entire grid
    entire_d_min = 0.5
    entire_d_max = 5.0
    entire_theta_min = -np.pi
    entire_theta_max = np.pi

    subdivide_square(entire_d_min, entire_d_max, entire_theta_min, entire_theta_max,
                     reachable_set_np, threshold_d, threshold_theta, final_squares)

    print(f"Number of final squares after subdivision: {len(final_squares)}")

    if len(final_squares) == 0:
        print("No squares were identified. Please check the subdivision logic.")
        return

    # Plotting using Rectangle patches
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    for (d_min, d_max, theta_min, theta_max, is_brt) in final_squares:
        # Define the color based on BRT membership
        if is_brt:
            color = 'red'
        else:
            color = 'white'  # Or any other color for non-BRT

        # Create a rectangle patch
        rect = patches.Rectangle(
            (theta_min, d_min),
            theta_max - theta_min,
            d_max - d_min,
            linewidth=0.5,
            edgecolor='black',
            facecolor=color,
            alpha=0.6 if is_brt else 0.3
        )
        ax.add_patch(rect)

    # Set plot limits
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0.5, 5.0)

    # Set labels and title
    ax.set_xlabel('Orientation Angle θ(t) [rad]')
    ax.set_ylabel('Distance to Obstacle Surface d(t)')
    ax.set_title('Backward Reachable Tube (BRT) Estimation with Adaptive Subdivision using MPPI')

    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='In BRT',
               markerfacecolor='red', markersize=15, alpha=0.6),
        Line2D([0], [0], marker='s', color='w', label='Not in BRT',
               markerfacecolor='white', markersize=15, markeredgecolor='black', alpha=0.3)
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()
