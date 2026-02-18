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
    
    hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible = data
    
    return hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible


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


def plot_positions(hist_t, hist_q, save_path=None):
    """Plot position trajectories for all agents."""
    num_agents = len(hist_q[0]) if len(hist_q) > 0 else 0
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['x', 'y', 'z']
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    for i in range(num_agents):
        q_agent = extract_agent_data(hist_q, i)
        for j, ax in enumerate(axes):
            ax.plot(hist_t, q_agent[:, j], label=f'Agent {i+1}', color=colors[i], alpha=0.7)
            ax.set_ylabel(f'Position {labels[j]} (m)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', ncol=min(num_agents, 6))
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Agent Positions vs Time')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + '_positions.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_velocities(hist_t, hist_dotq, save_path=None):
    """Plot velocity trajectories for all agents."""
    num_agents = len(hist_dotq[0]) if len(hist_dotq) > 0 else 0
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['x', 'y', 'z']
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    for i in range(num_agents):
        dotq_agent = extract_agent_data(hist_dotq, i)
        for j, ax in enumerate(axes):
            ax.plot(hist_t, dotq_agent[:, j], label=f'Agent {i+1}', color=colors[i], alpha=0.7)
            ax.set_ylabel(f'Velocity {labels[j]} (m/s)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', ncol=min(num_agents, 6))
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Agent Velocities vs Time')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + '_velocities.png', dpi=300, bbox_inches='tight')
    plt.show()


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
        plt.savefig(save_path + '_accelerations.png', dpi=300, bbox_inches='tight')
    plt.show()


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
        plt.savefig(save_path + '_min_distance.png', dpi=300, bbox_inches='tight')
    plt.show()


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
        plt.savefig(save_path + '_feasibility.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_3d_trajectories(hist_q, save_path=None):
    """Plot 3D trajectories of all agents."""
    num_agents = len(hist_q[0]) if len(hist_q) > 0 else 0
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(num_agents):
        q_agent = extract_agent_data(hist_q, i)
        ax.plot(q_agent[:, 0], q_agent[:, 1], q_agent[:, 2], 
                label=f'Agent {i+1}', color=colors[i], linewidth=2, alpha=0.7)
        # Mark start and end points
        ax.scatter(q_agent[0, 0], q_agent[0, 1], q_agent[0, 2], 
                  color=colors[i], s=100, marker='o', edgecolors='black', linewidths=2)
        ax.scatter(q_agent[-1, 0], q_agent[-1, 1], q_agent[-1, 2], 
                  color=colors[i], s=100, marker='s', edgecolors='black', linewidths=2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectories of All Agents')
    ax.legend(loc='best', ncol=min(num_agents, 6))
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path + '_3d_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_2d_projections(hist_q, save_path=None):
    """Plot 2D projections (xy, xz, yz) of trajectories."""
    num_agents = len(hist_q[0]) if len(hist_q) > 0 else 0
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    projections = [
        (0, 1, 'XY Projection', 'X (m)', 'Y (m)'),
        (0, 2, 'XZ Projection', 'X (m)', 'Z (m)'),
        (1, 2, 'YZ Projection', 'Y (m)', 'Z (m)')
    ]
    
    for idx, (ax, (i, j, title, xlabel, ylabel)) in enumerate(zip(axes, projections)):
        for agent_idx in range(num_agents):
            q_agent = extract_agent_data(hist_q, agent_idx)
            ax.plot(q_agent[:, i], q_agent[:, j], 
                   label=f'Agent {agent_idx+1}', color=colors[agent_idx], linewidth=2, alpha=0.7)
            # Mark start and end points
            ax.scatter(q_agent[0, i], q_agent[0, j], 
                      color=colors[agent_idx], s=50, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
            ax.scatter(q_agent[-1, i], q_agent[-1, j], 
                      color=colors[agent_idx], s=50, marker='s', edgecolors='black', linewidths=1.5, zorder=5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=min(num_agents, 4))
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + '_2d_projections.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_speed_profile(hist_t, hist_dotq, save_path=None):
    """Plot speed (magnitude of velocity) for all agents."""
    num_agents = len(hist_dotq[0]) if len(hist_dotq) > 0 else 0
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for i in range(num_agents):
        dotq_agent = extract_agent_data(hist_dotq, i)
        speed = np.linalg.norm(dotq_agent, axis=1)
        ax.plot(hist_t, speed, label=f'Agent {i+1}', color=colors[i], linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Agent Speed vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=min(num_agents, 6))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + '_speed.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_shapes(hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible):
    
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
    parser.add_argument('-s', '--save', action='store_true',
                       help='Save plots as PNG files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (default: same as data file)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (useful when only saving)')
    
    args = parser.parse_args()
    
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
        hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible = load_data(pkl_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # DEBUG
    print_shapes(hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible)
    input("Press Enter to continue...")

    # Determine save path
    save_path = None
    if args.save:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(pkl_file)
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(pkl_file))[0]
        save_path = os.path.join(output_dir, base_name)
    
    # Set matplotlib backend if not showing
    if args.no_show:
        plt.ioff()
        import matplotlib
        matplotlib.use('Agg')
    
    print("Generating plots...")
    
    # Generate all plots
    plot_positions(hist_t, hist_q, save_path)
    plot_velocities(hist_t, hist_dotq, save_path)
    plot_accelerations(hist_t, hist_a, save_path)
    plot_min_distance(hist_t, hist_min_dist, save_path)
    plot_feasibility(hist_t, hist_feasible, save_path)
    plot_3d_trajectories(hist_q, save_path)
    plot_2d_projections(hist_q, save_path)
    plot_speed_profile(hist_t, hist_dotq, save_path)
    
    if save_path:
        print(f"Plots saved to: {output_dir}")
    
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
