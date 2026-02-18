import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D


def load_data(pkl_file):
    """Load data from pickle file."""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible, q_tg = data
    
    return hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible, q_tg


def extract_agent_data(hist_list, agent_idx):
    """Extract data for a specific agent from history list."""
    agent_data = []
    for timestep in hist_list:
        if len(timestep) > agent_idx:
            # Convert matrix to array and extract values
            if isinstance(timestep[agent_idx], np.matrix):
                agent_data.append(np.array(timestep[agent_idx]).flatten())
            else:
                agent_data.append(np.array(timestep[agent_idx]).flatten())
    return np.array(agent_data)


def extract_target_data(q_tg, agent_idx, hist_t=None):
    """Extract target data for a specific agent from q_tg.
    
    q_tg structure: q_tg[timestep][agent] = matrix (target position)
    
    Parameters
    ----------
    q_tg : list
        Target position history (may have fewer points than simulation timesteps)
    agent_idx : int
        Agent index to extract
    hist_t : list, optional
        Time history for interpolation. If provided and q_tg has fewer points,
        target positions will be interpolated to match simulation timesteps.
    
    Returns
    -------
    agent_data : numpy array
        Array of target positions for the agent, shape (n_timesteps, 3)
    """
    agent_data = []
    for timestep in q_tg:
        if isinstance(timestep, list) and len(timestep) > agent_idx:
            # Extract target position for this agent at this timestep
            target_pos = timestep[agent_idx]
            if isinstance(target_pos, np.matrix):
                agent_data.append(np.array(target_pos).flatten())
            elif isinstance(target_pos, np.ndarray):
                agent_data.append(target_pos.flatten())
            elif isinstance(target_pos, list):
                agent_data.append(np.array(target_pos).flatten())
            else:
                # Try to convert to array
                agent_data.append(np.array(target_pos).flatten())
    
    agent_data = np.array(agent_data) if len(agent_data) > 0 else np.array([]).reshape(0, 3)
    
    # If hist_t is provided and has more points than q_tg, interpolate
    if hist_t is not None and len(hist_t) > len(agent_data) and len(agent_data) > 0:
        # The target curve is parameterized from 0 to 1
        # Map simulation time to curve parameter space
        t_curve = np.linspace(0, 1, len(agent_data))
        t_sim = np.array(hist_t)
        
        # Normalize simulation time to [0, 1] if needed
        if t_sim.max() > 1.0:
            t_sim_norm = (t_sim - t_sim.min()) / (t_sim.max() - t_sim.min())
        else:
            t_sim_norm = t_sim
        
        # Interpolate each component using numpy
        interpolated = np.zeros((len(hist_t), 3))
        for i in range(3):
            # Use numpy's interp function for linear interpolation
            interpolated[:, i] = np.interp(t_sim_norm, t_curve, agent_data[:, i])
        
        agent_data = interpolated
    
    return agent_data


def calculate_position_errors(hist_q, q_tg, hist_t=None, agent_indices=None, verbose=False):
    """Calculate position errors between actual positions and target curves.
    
    Parameters
    ----------
    hist_q : list
        Actual position history for all agents.
    q_tg : list
        Target curve samples for all agents (q_tg[sample_idx][agent_idx] is a 3D point).
    agent_indices : list, optional
        List of agent indices to calculate errors for
    verbose : bool, optional
        If True, print diagnostic information about data lengths
    
    Returns
    -------
    errors : dict
        Dictionary with keys:
        - 'magnitude': list of arrays, one per agent, containing minimum distance to the agent's target curve.
        - 'components': list of arrays, one per agent, containing the vector from closest target-curve point to actual position.
        - 'agent_indices': list of agent indices processed
    """
    num_agents = len(hist_q[0]) if len(hist_q) > 0 else 0
    
    if agent_indices is None:
        agent_indices = list(range(num_agents))
    
    # Filter to valid agent indices
    agent_indices = [i for i in agent_indices if 0 <= i < num_agents]
    
    errors = {
        'magnitude': [],
        'components': [],
        'agent_indices': agent_indices
    }
    
    if verbose:
        print(f"\nPosition Error Calculation Diagnostics:")
        print(f"  hist_q length: {len(hist_q)}")
        print(f"  q_tg length: {len(q_tg)}")
    
    for agent_idx in agent_indices:
        q_actual = extract_agent_data(hist_q, agent_idx)
        q_curve = extract_target_data(q_tg, agent_idx)

        if verbose:
            print(f"  Agent {agent_idx}: q_actual length = {len(q_actual)}, curve samples = {len(q_curve)}")

        if len(q_actual) == 0 or len(q_curve) == 0:
            if verbose:
                print("    WARNING: Empty actual trajectory or empty target curve.")
            errors['components'].append(np.array([]).reshape(0, 3))
            errors['magnitude'].append(np.array([]))
            continue

        # Compute nearest-point-to-curve error for every actual sample in chunks
        # to avoid very large temporary arrays for long simulations.
        chunk_size = 500
        error_components_chunks = []
        error_magnitude_chunks = []

        for start in range(0, len(q_actual), chunk_size):
            end = min(start + chunk_size, len(q_actual))
            q_chunk = q_actual[start:end]  # (C, 3)

            # diff[c, m, :] = q_chunk[c, :] - q_curve[m, :]
            diff = q_chunk[:, np.newaxis, :] - q_curve[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)  # (C, M)
            min_idx = np.argmin(dist_sq, axis=1)
            closest_curve_points = q_curve[min_idx]  # (C, 3)

            err_vec = q_chunk - closest_curve_points
            err_mag = np.sqrt(np.min(dist_sq, axis=1))

            error_components_chunks.append(err_vec)
            error_magnitude_chunks.append(err_mag)

        errors['components'].append(np.vstack(error_components_chunks))
        errors['magnitude'].append(np.concatenate(error_magnitude_chunks))
    
    return errors


def extract_acceleration_data(hist_a, agent_idx):
    """Extract acceleration data for a specific agent.
    
    The acceleration data structure: each timestep contains a list of 3 scalars
    (x, y, z components), where each scalar is a 1x1 matrix.
    Due to a bug in paper_icuas.py line 249, only one agent's data is stored.
    Structure: hist_a[timestep] = [ax_matrix, ay_matrix, az_matrix]
    """
    agent_data = []
    for timestep in hist_a:
        if isinstance(timestep, list):
            if len(timestep) == 3:
                # Structure: [ax, ay, az] where each is a 1x1 matrix
                # This is for a single agent (agent_idx should be 0)
                if agent_idx == 0:
                    components = []
                    for comp in timestep:
                        if isinstance(comp, np.matrix):
                            components.append(float(comp.item()))
                        elif isinstance(comp, np.ndarray):
                            components.append(float(comp.item()))
                        else:
                            components.append(float(comp))
                    agent_data.append(components)
            elif len(timestep) > agent_idx:
                # Structure: timestep is a list of agents
                agent_accel = timestep[agent_idx]
                if isinstance(agent_accel, list) and len(agent_accel) == 3:
                    # Extract the 3 components (each is a 1x1 matrix)
                    components = []
                    for comp in agent_accel:
                        if isinstance(comp, np.matrix):
                            components.append(float(comp.item()))
                        else:
                            components.append(float(comp))
                    agent_data.append(components)
                elif isinstance(agent_accel, (np.matrix, np.ndarray)):
                    # If it's already a vector, flatten it
                    arr = np.array(agent_accel).flatten()
                    if len(arr) >= 3:
                        agent_data.append(arr[:3].tolist())
                    else:
                        # Pad with zeros if needed
                        padded = np.zeros(3)
                        padded[:len(arr)] = arr
                        agent_data.append(padded.tolist())
    
    if len(agent_data) == 0:
        return np.array([]).reshape(0, 3)
    
    return np.array(agent_data)


def plot_positions(hist_t, hist_q, agent_indices=None, save_path=None):
    """Plot position trajectories for selected agents."""
    num_agents = len(hist_q[0]) if len(hist_q) > 0 else 0
    
    if agent_indices is None:
        agent_indices = list(range(num_agents))
    
    # Filter to valid agent indices
    agent_indices = [i for i in agent_indices if 0 <= i < num_agents]
    
    if len(agent_indices) == 0:
        print("Warning: No valid agents to plot.")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['x', 'y', 'z']
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    for i in agent_indices:
        q_agent = extract_agent_data(hist_q, i)
        for j, ax in enumerate(axes):
            ax.plot(hist_t, q_agent[:, j], label=f'Agent {i+1}', color=colors[i], alpha=0.7)
            ax.set_ylabel(f'Position {labels[j]} (m)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', ncol=min(len(agent_indices), 6), fontsize=9, framealpha=0.9)
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Agent Positions vs Time')
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'positions.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path + '_positions.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def plot_velocities(hist_t, hist_dotq, agent_indices=None, save_path=None):
    """Plot velocity trajectories for selected agents."""
    num_agents = len(hist_dotq[0]) if len(hist_dotq) > 0 else 0
    
    if agent_indices is None:
        agent_indices = list(range(num_agents))
    
    # Filter to valid agent indices
    agent_indices = [i for i in agent_indices if 0 <= i < num_agents]
    
    if len(agent_indices) == 0:
        print("Warning: No valid agents to plot.")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['x', 'y', 'z']
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    for i in agent_indices:
        dotq_agent = extract_agent_data(hist_dotq, i)
        for j, ax in enumerate(axes):
            ax.plot(hist_t, dotq_agent[:, j], label=f'Agent {i+1}', color=colors[i], alpha=0.7)
            ax.set_ylabel(f'Velocity {labels[j]} (m/s)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', ncol=min(len(agent_indices), 6), fontsize=9, framealpha=0.9)
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Agent Velocities vs Time')
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'velocities.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path + '_velocities.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def plot_accelerations(hist_t, hist_a, save_path=None):
    """Plot acceleration trajectories for all agents."""
    if len(hist_a) == 0:
        print("Warning: No acceleration data found. Skipping acceleration plot.")
        return
    
    # Determine number of agents from the data structure
    # Check the structure: hist_a[0] should be a list
    if not isinstance(hist_a[0], list):
        print("Warning: Acceleration data structure may be incorrect. Skipping acceleration plot.")
        return
    
    # Check the structure
    # If hist_a[0] has 3 elements and each is a matrix/array, it's likely [ax, ay, az] for one agent
    # If hist_a[0] has more elements, each might be an agent's data
    if len(hist_a[0]) == 3 and all(isinstance(hist_a[0][i], (np.matrix, np.ndarray)) for i in range(3)):
        # Structure: hist_a[timestep] = [ax, ay, az] for a single agent (due to bug in paper_icuas.py)
        print("Note: Acceleration data appears to be for a single agent (likely due to bug in data collection).")
        num_agents = 1
    elif len(hist_a[0]) > 3:
        # Structure: hist_a[timestep] = [agent0_data, agent1_data, ...]
        num_agents = len(hist_a[0])
    else:
        # Try to determine from first element structure
        if isinstance(hist_a[0][0], list) and len(hist_a[0][0]) == 3:
            num_agents = len(hist_a[0])
        else:
            print("Warning: Could not determine acceleration data structure. Trying with 1 agent.")
            num_agents = 1
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['x', 'y', 'z']
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    for i in range(num_agents):
        a_agent = extract_acceleration_data(hist_a, i)
        
        # Check if we got valid data
        if len(a_agent) == 0:
            print(f"Warning: No acceleration data found for agent {i+1}. Skipping.")
            continue
        
        # Ensure we have the right shape
        if len(a_agent.shape) < 2 or a_agent.shape[1] < 3:
            print(f"Warning: Agent {i+1} acceleration data has unexpected shape {a_agent.shape}. Skipping.")
            continue
        
        # Ensure time array matches data length
        time_data = hist_t[:len(a_agent)]
        
        for j, ax in enumerate(axes):
            ax.plot(time_data, a_agent[:, j], label=f'Agent {i+1}', color=colors[i], alpha=0.7)
            ax.set_ylabel(f'Acceleration {labels[j]} (m/s²)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', ncol=min(num_agents, 6))
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Agent Accelerations vs Time')
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'accelerations.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path + '_accelerations.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def plot_min_distance(hist_t, hist_min_dist, save_path=None):
    """Plot minimum distance to obstacles over time."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(hist_t, hist_min_dist, 'r-', linewidth=2, label='Min distance to obstacles')
    ax.axhline(y=0.05, color='orange', linestyle='--', linewidth=1.5, label='Safety threshold (0.05 m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Minimum Distance (m)')
    ax.set_title('Minimum Distance to Obstacles vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'min_distance.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path + '_min_distance.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def plot_feasibility(hist_t, hist_feasible, save_path=None):
    """Plot feasibility flag over time."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Convert boolean to int for plotting
    feasible_int = [1 if f else 0 for f in hist_feasible]
    
    ax.fill_between(hist_t, 0, feasible_int, alpha=0.5, color='green', label='Feasible')
    ax.plot(hist_t, feasible_int, 'g-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Feasibility')
    ax.set_title('Solution Feasibility vs Time')
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Infeasible', 'Feasible'])
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'feasibility.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path + '_feasibility.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def plot_3d_trajectories(hist_q, q_tg, agent_indices=None, save_path=None):
    """Plot 3D trajectories of selected agents with target trajectories.
    
    Parameters
    ----------
    hist_q : list
        Position history for all agents
    q_tg : list
        Target position history for all agents
    agent_indices : list, optional
        List of agent indices to plot (0-based). If None, plots all agents.
    save_path : str, optional
        Path to save the plot
    """
    num_agents = len(hist_q[0]) if len(hist_q) > 0 else 0
    
    if agent_indices is None:
        agent_indices = list(range(num_agents))
    
    # Filter to valid agent indices
    agent_indices = [i for i in agent_indices if 0 <= i < num_agents]
    
    if len(agent_indices) == 0:
        print("Warning: No valid agents to plot.")
        return
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in agent_indices:
        q_agent = extract_agent_data(hist_q, i)
        
        # Plot actual trajectory
        ax.plot(q_agent[:, 0], q_agent[:, 1], q_agent[:, 2], 
                label=f'Agent {i+1}', color=colors[i], linewidth=2, alpha=0.7)
        
        # Extract and plot target trajectory (dashed, dimmer)
        # Note: hist_t is not available here, so we'll extract without interpolation
        # The 3D plot will handle length mismatches
        q_target = extract_target_data(q_tg, i)
        if len(q_target) > 0:
            # Ensure same length as actual trajectory for proper comparison
            min_len = min(len(q_agent), len(q_target))
            q_target_plot = q_target[:min_len]
            
            ax.plot(q_target_plot[:, 0], q_target_plot[:, 1], q_target_plot[:, 2], 
                   '--', color=colors[i], linewidth=1.5, alpha=0.3, label=f'Agent {i+1} target')
        
        # Mark start and end points
        ax.scatter(q_agent[0, 0], q_agent[0, 1], q_agent[0, 2], 
                  color=colors[i], s=100, marker='o', edgecolors='black', linewidths=2)
        ax.scatter(q_agent[-1, 0], q_agent[-1, 1], q_agent[-1, 2], 
                  color=colors[i], s=100, marker='s', edgecolors='black', linewidths=2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectories of Selected Agents')
    ax.legend(loc='best', ncol=min(len(agent_indices), 6), fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        if os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, '3d_trajectories.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path + '_3d_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def plot_2d_projections(hist_q, q_tg, agent_indices=None, save_path=None):
    """Plot 2D projections (xy, xz, yz) of trajectories with target trajectories.
    Creates 3 separate figures, one for each projection.
    
    Parameters
    ----------
    hist_q : list
        Position history for all agents
    q_tg : list
        Target position history for all agents
    agent_indices : list, optional
        List of agent indices to plot (0-based). If None, plots all agents.
    save_path : str, optional
        Path to save the plots
    """
    num_agents = len(hist_q[0]) if len(hist_q) > 0 else 0
    
    if agent_indices is None:
        agent_indices = list(range(num_agents))
    
    # Filter to valid agent indices
    agent_indices = [i for i in agent_indices if 0 <= i < num_agents]
    
    if len(agent_indices) == 0:
        print("Warning: No valid agents to plot.")
        return
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    projections = [
        (0, 1, 'XY Projection', 'X (m)', 'Y (m)', 'xy'),
        (0, 2, 'XZ Projection', 'X (m)', 'Z (m)', 'xz'),
        (1, 2, 'YZ Projection', 'Y (m)', 'Z (m)', 'yz')
    ]
    
    # Create a separate figure for each projection
    for i, j, title, xlabel, ylabel, suffix in projections:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        for agent_idx in agent_indices:
            q_agent = extract_agent_data(hist_q, agent_idx)
            
            # Plot actual trajectory
            ax.plot(q_agent[:, i], q_agent[:, j], 
                   label=f'Agent {agent_idx+1}', color=colors[agent_idx], linewidth=2, alpha=0.7)
            
            # Extract and plot target trajectory (dashed, dimmer)
            q_target = extract_target_data(q_tg, agent_idx)
            if len(q_target) > 0:
                # Ensure same length as actual trajectory for proper comparison
                min_len = min(len(q_agent), len(q_target))
                q_target_plot = q_target[:min_len]
                
                ax.plot(q_target_plot[:, i], q_target_plot[:, j], 
                       '--', color=colors[agent_idx], linewidth=1.5, alpha=0.3, 
                       label=f'Agent {agent_idx+1} target')
            
            # Mark start and end points
            ax.scatter(q_agent[0, i], q_agent[0, j], 
                      color=colors[agent_idx], s=50, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
            ax.scatter(q_agent[-1, i], q_agent[-1, j], 
                      color=colors[agent_idx], s=50, marker='s', edgecolors='black', linewidths=1.5, zorder=5)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Compact legend: smaller font, fewer columns, placed outside plot area
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                          ncol=1, fontsize=9, framealpha=0.9, 
                          columnspacing=0.5, handlelength=1.5)
        legend.get_frame().set_linewidth(0.5)
        
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        
        if save_path:
            if os.path.isdir(save_path):
                plt.savefig(os.path.join(save_path, f'2d_projection_{suffix}.png'), dpi=300, bbox_inches='tight')
            else:
                plt.savefig(save_path + f'_2d_projection_{suffix}.png', dpi=300, bbox_inches='tight')
        
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to ensure plot is rendered


def plot_speed_profile(hist_t, hist_dotq, agent_indices=None, save_path=None):
    """Plot speed (magnitude of velocity) for selected agents."""
    num_agents = len(hist_dotq[0]) if len(hist_dotq) > 0 else 0
    
    if agent_indices is None:
        agent_indices = list(range(num_agents))
    
    # Filter to valid agent indices
    agent_indices = [i for i in agent_indices if 0 <= i < num_agents]
    
    if len(agent_indices) == 0:
        print("Warning: No valid agents to plot.")
        return
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for i in agent_indices:
        dotq_agent = extract_agent_data(hist_dotq, i)
        speed = np.linalg.norm(dotq_agent, axis=1)
        ax.plot(hist_t, speed, label=f'Agent {i+1}', color=colors[i], linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Agent Speed vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=min(len(agent_indices), 6))
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'speed.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path + '_speed.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def plot_position_errors(hist_t, hist_q, q_tg, agent_indices=None, save_path=None, verbose=False):
    """Plot position errors between actual and target positions.
    
    Parameters
    ----------
    hist_t : list
        Time history
    hist_q : list
        Actual position history for all agents
    q_tg : list
        Target position history for all agents
    agent_indices : list, optional
        List of agent indices to plot (0-based). If None, plots all agents.
    save_path : str, optional
        Path to save the plot
    verbose : bool, optional
        If True, print diagnostic information about data lengths
    """
    # Calculate position errors
    errors = calculate_position_errors(hist_q, q_tg, hist_t=hist_t, agent_indices=agent_indices, verbose=verbose)
    
    if verbose:
        print(f"\nPlotting Diagnostics:")
        print(f"  hist_t length: {len(hist_t)}")
        print(f"  hist_t range: {min(hist_t):.2f} to {max(hist_t):.2f} seconds")
        if len(errors['magnitude']) > 0:
            print(f"  Error data length (agent 0): {len(errors['magnitude'][0])}")
            if len(hist_t) > 0:
                expected_time = hist_t[len(errors['magnitude'][0]) - 1] if len(errors['magnitude'][0]) <= len(hist_t) else hist_t[-1]
                print(f"  Expected time range: 0 to {expected_time:.2f} seconds")
    
    if len(errors['magnitude']) == 0:
        print("Warning: No position error data to plot.")
        return
    
    num_agents_total = len(hist_q[0]) if len(hist_q) > 0 else 0
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents_total))
    
    # Plot 1: Magnitude of position error
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    for idx, agent_idx in enumerate(errors['agent_indices']):
        error_mag = errors['magnitude'][idx]
        # Ensure time array matches error length
        time_data = hist_t[:len(error_mag)]
        ax1.plot(time_data, error_mag, label=f'Agent {agent_idx+1}', 
                color=colors[agent_idx], linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position Error Magnitude (m)')
    ax1.set_title('Position Error Magnitude vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', ncol=min(len(errors['agent_indices']), 6), fontsize=9, framealpha=0.9)
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'position_error_magnitude.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path + '_position_error_magnitude.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)
    
    # Plot 2: Component-wise position errors
    fig2, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['x', 'y', 'z']
    
    for idx, agent_idx in enumerate(errors['agent_indices']):
        error_comp = errors['components'][idx]
        time_data = hist_t[:len(error_comp)]
        
        for j, ax in enumerate(axes):
            ax.plot(time_data, error_comp[:, j], label=f'Agent {agent_idx+1}', 
                   color=colors[agent_idx], linewidth=2, alpha=0.7)
            ax.set_ylabel(f'Position Error {labels[j]} (m)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', ncol=min(len(errors['agent_indices']), 6), fontsize=9, framealpha=0.9)
            # Add zero line for reference
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Position Error Components vs Time')
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'position_error_components.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path + '_position_error_components.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def print_shapes(hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible, q_tg):
    
    # Debug: Print shapes of all data
    print("\n" + "="*60)
    print("DEBUG: Data Shapes")
    print("="*60)
    print(f"hist_t shape: {np.array(hist_t).shape}")
    print(f"hist_t type: {type(hist_t)}")
    print(f"hist_t length: {len(hist_t)}")
    if len(hist_t) > 0:
        print(f"hist_t[0] type: {type(hist_t[0])}, value: {hist_t[0]}")
    
    print(f"\nhist_q length: {len(hist_q)}")
    if len(hist_q) > 0:
        print(f"hist_q[0] type: {type(hist_q[0])}, length: {len(hist_q[0])}")
        if len(hist_q[0]) > 0:
            print(f"hist_q[0][0] type: {type(hist_q[0][0])}")
            if isinstance(hist_q[0][0], np.matrix):
                print(f"hist_q[0][0] shape: {hist_q[0][0].shape}")
            else:
                print(f"hist_q[0][0] shape: {np.array(hist_q[0][0]).shape}")
            print(f"hist_q[0][0] value:\n{hist_q[0][0]}")
    
    print(f"\nhist_dotq length: {len(hist_dotq)}")
    if len(hist_dotq) > 0:
        print(f"hist_dotq[0] type: {type(hist_dotq[0])}, length: {len(hist_dotq[0])}")
        if len(hist_dotq[0]) > 0:
            print(f"hist_dotq[0][0] type: {type(hist_dotq[0][0])}")
            if isinstance(hist_dotq[0][0], np.matrix):
                print(f"hist_dotq[0][0] shape: {hist_dotq[0][0].shape}")
            else:
                print(f"hist_dotq[0][0] shape: {np.array(hist_dotq[0][0]).shape}")
            print(f"hist_dotq[0][0] value:\n{hist_dotq[0][0]}")
    
    print(f"\nhist_a length: {len(hist_a)}")
    if len(hist_a) > 0:
        print(f"hist_a[0] type: {type(hist_a[0])}, length: {len(hist_a[0]) if isinstance(hist_a[0], (list, np.ndarray)) else 'N/A'}")
        if isinstance(hist_a[0], list) and len(hist_a[0]) > 0:
            print(f"hist_a[0][0] type: {type(hist_a[0][0])}")
            if isinstance(hist_a[0][0], np.matrix):
                print(f"hist_a[0][0] shape: {hist_a[0][0].shape}")
            else:
                print(f"hist_a[0][0] shape: {np.array(hist_a[0][0]).shape}")
            print(f"hist_a[0][0] value:\n{hist_a[0][0]}")
        elif isinstance(hist_a[0], np.ndarray):
            print(f"hist_a[0] shape: {hist_a[0].shape}")
            print(f"hist_a[0] value:\n{hist_a[0]}")
        else:
            print(f"hist_a[0] value: {hist_a[0]}")
    
    print(f"\nhist_min_dist length: {len(hist_min_dist)}")
    if len(hist_min_dist) > 0:
        print(f"hist_min_dist[0] type: {type(hist_min_dist[0])}, value: {hist_min_dist[0]}")
    
    print(f"\nhist_feasible length: {len(hist_feasible)}")
    if len(hist_feasible) > 0:
        print(f"hist_feasible[0] type: {type(hist_feasible[0])}, value: {hist_feasible[0]}")
    
    print(f"\nq_tg length: {len(q_tg)}")
    if len(q_tg) > 0:
        print(f"q_tg[0] type: {type(q_tg[0])}, value: {q_tg[0]}")
    
    # Test extract_agent_data for accelerations
    if len(hist_a) > 0:
        print(f"\nTesting extract_agent_data for accelerations:")
        try:
            test_a = extract_agent_data(hist_a, 0)
            print(f"  Agent 0 acceleration data shape: {test_a.shape}")
            print(f"  Agent 0 acceleration data:\n{test_a[:5] if len(test_a) > 5 else test_a}")
        except Exception as e:
            print(f"  Error extracting agent 0 acceleration: {e}")
    
    print("="*60 + "\n")
    

def main():
    parser = argparse.ArgumentParser(description='Plot data from ICAS simulation pickle files')
    parser.add_argument('-f', '--file', type=str, required=True,
                       help='Path to the .pkl file (relative to data/ directory or full path)')
    parser.add_argument('-n', '--no-save', action='store_true',
                       help='Do not save plots as PNG files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (default: same as data file)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (useful when only saving)')
    parser.add_argument('--agents', type=str, default='0,1,2',
                       help='Comma-separated list of agent indices to plot (0-based, default: 0,1,2)')
    
    args = parser.parse_args()
    
    # Parse agent indices
    try:
        agent_indices = [int(x.strip()) for x in args.agents.split(',')]
    except ValueError:
        print(f"Error: Invalid agent indices format: {args.agents}. Use comma-separated integers (e.g., '0,1,2').")
        return
    
    # Determine file path
    if os.path.isabs(args.file):
        pkl_file = args.file
    else:
        # Try relative to current directory first
        if os.path.exists(args.file):
            pkl_file = args.file
        else:
            # Try in data/ directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data')
            pkl_file = os.path.join(data_dir, args.file)
    
    if not os.path.exists(pkl_file):
        print(f"Error: File not found: {pkl_file}")
        return
    
    print(f"Loading data from: {pkl_file}")
    
    # Load data
    try:
        hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible, q_tg = load_data(pkl_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # DEBUG
    print_shapes(hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible, q_tg)

    # Determine save path
    save_path = None
    if not args.no_save:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(pkl_file)
        
        save_path = os.path.join(output_dir, "plots")
        os.makedirs(save_path, exist_ok=True)
    
    print(f"Save path: {save_path}")

    # Set matplotlib backend if not showing
    if args.no_show:
        plt.ioff()
        import matplotlib
        matplotlib.use('Agg')
    
    # Determine number of agents
    num_agents = len(hist_q[0]) if len(hist_q) > 0 else 0
    
    # Filter agent indices to valid range
    agent_indices = [i for i in agent_indices if 0 <= i < num_agents]
    
    if len(agent_indices) == 0:
        print(f"Error: No valid agent indices. Available agents: 0 to {num_agents-1}")
        return
    
    print(f"Plotting agents: {agent_indices}")
    print("Generating plots...")
    
    # Generate all plots
    plot_positions(hist_t, hist_q, agent_indices, save_path)
    plot_velocities(hist_t, hist_dotq, agent_indices, save_path)
    # plot_accelerations(hist_t, hist_a, save_path)
    plot_min_distance(hist_t, hist_min_dist, save_path)
    # plot_feasibility(hist_t, hist_feasible, save_path)
    plot_3d_trajectories(hist_q, q_tg, agent_indices, save_path)
    plot_2d_projections(hist_q, q_tg, agent_indices, save_path)
    plot_speed_profile(hist_t, hist_dotq, agent_indices, save_path)
    plot_position_errors(hist_t, hist_q, q_tg, agent_indices, save_path, verbose=True)
    
    if save_path:
        print(f"Plots saved to: {output_dir}")
    
    if not args.no_show:
        # Keep plots open - show all at once
        plt.show(block=True)


if __name__ == "__main__":
    main()
