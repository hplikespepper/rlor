# #!/usr/bin/env python3
# """
# 可拼车CVRP问题可视化脚本 - 带箭头版本
# """

# import numpy as np
# import torch
# import gym
# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyArrowPatch
# import argparse

# from models.attention_model_wrapper import Agent
# from wrappers.syncVectorEnvPomo import SyncVectorEnv
# from wrappers.recordWrapper import RecordEpisodeStatistics
# import os


# def plot_solution(nodes_coordinates, trajectory, title="Ride-sharing CVRP Solution"):
#     """Plot CVRP solution with arrows showing direction"""
#     plt.figure(figsize=(12, 10))
    
#     # Calculate number of nodes
#     total_nodes = len(nodes_coordinates)
#     max_nodes = (total_nodes - 1) // 2
    
#     # Plot depot
#     depot_coord = nodes_coordinates[0]
#     plt.scatter(depot_coord[0], depot_coord[1], s=300, c='black', marker='s', 
#                 label='Depot', zorder=5, edgecolors='white', linewidth=2)
    
#     # Plot pickup points
#     if total_nodes > 1:
#         pickup_coords = nodes_coordinates[1:max_nodes+1]
#         plt.scatter(pickup_coords[:, 0], pickup_coords[:, 1], s=150, c='forestgreen', 
#                     marker='o', label='Pickup Points', zorder=4, alpha=0.8)
    
#     # Plot dropoff points
#     if total_nodes > max_nodes + 1:
#         dropoff_coords = nodes_coordinates[max_nodes+1:2*max_nodes+1]
#         plt.scatter(dropoff_coords[:, 0], dropoff_coords[:, 1], s=150, c='crimson', 
#                     marker='^', label='Dropoff Points', zorder=4, alpha=0.8)
    
#     # Plot path with arrows
#     path_coords = nodes_coordinates[trajectory]
#     total_distance = 0.0
    
#     # Generate colors for different segments (lighter to darker gray)
#     num_segments = len(trajectory) - 1
#     colors = plt.cm.Greys(np.linspace(0.4, 0.9, num_segments))
    
#     for i in range(len(trajectory) - 1):
#         start_coord = path_coords[i]
#         end_coord = path_coords[i + 1]
        
#         # Calculate distance
#         dist = np.sqrt(np.sum((start_coord - end_coord) ** 2))
#         total_distance += dist
        
#         # Draw arrow
#         arrow = FancyArrowPatch(
#             start_coord, end_coord,
#             arrowstyle='->', 
#             mutation_scale=20,
#             linewidth=2.5,
#             color=colors[i],
#             alpha=0.8,
#             zorder=3
#         )
#         plt.gca().add_patch(arrow)
    
#     plt.axis('equal')
    
#     # Add node labels
#     for i, coord in enumerate(nodes_coordinates):
#         if i == 0:
#             plt.annotate('Depot', (coord[0], coord[1]), xytext=(5, 5), 
#                         textcoords='offset points', fontsize=14, fontweight='bold')
#         elif i <= max_nodes:
#             plt.annotate(f'P{i}', (coord[0], coord[1]), xytext=(5, 5), 
#                         textcoords='offset points', fontsize=14)
#         else:
#             plt.annotate(f'D{i-max_nodes}', (coord[0], coord[1]), xytext=(5, 5), 
#                         textcoords='offset points', fontsize=14)

#     plt.title(title, fontsize=16, fontweight='bold')
#     plt.xlabel('X', fontsize=16)
#     plt.ylabel('Y', fontsize=16)
#     plt.legend(loc='best', fontsize=14)
    
#     plt.text(0.98, 0.02, f'Total Distance: {total_distance:.3f}', 
#              transform=plt.gca().transAxes, fontsize=16, 
#              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
#              verticalalignment='bottom', horizontalalignment='right')
    
#     plt.tight_layout()
#     return total_distance


# def make_env(env_id, seed, cfg={}):
#     """创建环境"""
#     def thunk():
#         env = gym.make(env_id, **cfg)
#         env = RecordEpisodeStatistics(env)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env
#     return thunk


# def solve_and_visualize(model_path, max_nodes=10, capacity_limit=20, n_traj=50, device='cuda', seed=42):
#     """加载模型并求解CVRP问题"""
    
#     # 注册环境
#     env_id = 'cvrp-v0'
#     env_entry_point = 'envs.cvrp_vector_env:CVRPVectorEnv'
    
#     try:
#         gym.envs.register(id=env_id, entry_point=env_entry_point)
#     except:
#         pass
    
#     # 创建环境
#     envs = SyncVectorEnv([
#         make_env(
#             env_id, 
#             seed, 
#             cfg={
#                 'max_nodes': max_nodes,
#                 'capacity_limit': capacity_limit,
#                 'n_traj': n_traj,
#                 'eval_data': False
#             }
#         )
#     ])
    
#     # 创建并加载模型
#     device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
#     agent = Agent(device=device, name='cvrp').to(device)
#     agent.load_state_dict(torch.load(model_path, map_location=device))
#     agent.eval()
    
#     print(f"Using device: {device}")
#     print("Starting CVRP solving...")
    
#     # Solving process
#     obs = envs.reset()
#     done = np.array([False])
#     trajectories = []
    
#     while not done.all():
#         with torch.no_grad():
#             action, logits = agent(obs)
        
#         # Multi-greedy inference: first step selects different starting pickup points
#         if len(trajectories) == 0:
#             pickup_actions = torch.arange(1, max_nodes + 1)
#             if n_traj <= max_nodes:
#                 action = pickup_actions[:n_traj].repeat(1, 1)
#             else:
#                 repeated_actions = pickup_actions.repeat((n_traj // max_nodes) + 1)
#                 action = repeated_actions[:n_traj].repeat(1, 1)
        
#         obs, reward, done, info = envs.step(action.cpu().numpy())
#         trajectories.append(action.cpu().numpy())
    
#     print(f"Solving completed in {len(trajectories)} steps")
    
#     # Get optimal solution
#     nodes_coordinates = np.vstack([obs['depot'], obs['observations'][0]])
#     final_return = info[0]['episode']['r']
#     best_traj = np.argmax(final_return)
#     resulting_traj = np.array(trajectories)[:, 0, best_traj]
#     resulting_traj_with_depot = np.hstack([np.zeros(1, dtype=int), resulting_traj])
    
#     best_distance = -final_return[best_traj]
    
#     print(f"Optimal path length: {best_distance:.4f}")
#     print(f"Path: {resulting_traj_with_depot}")
    
#     caption = f"Ride-sharing CVRP Solution: seed={seed}"
#     # Generate visualization
#     total_distance = plot_solution(
#         nodes_coordinates, 
#         resulting_traj_with_depot,
#         title=caption
#     )
    
#     return total_distance


# def main():
#     parser = argparse.ArgumentParser(description='Ride-sharing CVRP Visualization')
#     parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
#     parser.add_argument('--max-nodes', type=int, default=10, help='Number of pickup points')
#     parser.add_argument('--capacity-limit', type=int, default=20, help='Vehicle capacity limit')
#     parser.add_argument('--n-traj', type=int, default=50, help='Number of POMO trajectories')
#     parser.add_argument('--device', type=str, default='cuda', help='Computing device')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
#     parser.add_argument('--save-dir', type=str, default='visualization', help='Path to save image')
    
#     args = parser.parse_args()
    
#     # Solve and visualize
#     total_distance = solve_and_visualize(
#         args.model_path, 
#         args.max_nodes, 
#         args.capacity_limit, 
#         args.n_traj, 
#         args.device, 
#         args.seed
#     )
#     # Save image
#     save_path = os.path.join(args.save_dir, f"{args.seed}.png")
#     os.makedirs(args.save_dir, exist_ok=True)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Image saved to: {save_path}")

#     # Show image
#     plt.show()


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
可拼车CVRP问题可视化脚本 - 修复版本
"""

import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import argparse

from models.attention_model_wrapper import Agent
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics
import os


def plot_solution(nodes_coordinates, trajectory, title="Ride-sharing CVRP Solution"):
    """Plot CVRP solution with arrows showing direction"""
    plt.figure(figsize=(12, 10))
    
    # Calculate number of nodes
    total_nodes = len(nodes_coordinates)
    max_nodes = (total_nodes - 1) // 2
    
    # Plot depot
    depot_coord = nodes_coordinates[0]
    plt.scatter(depot_coord[0], depot_coord[1], s=300, c='black', marker='s', 
                label='Depot', zorder=5, edgecolors='white', linewidth=2)
    
    # Plot pickup points
    if total_nodes > 1:
        pickup_coords = nodes_coordinates[1:max_nodes+1]
        plt.scatter(pickup_coords[:, 0], pickup_coords[:, 1], s=150, c='forestgreen', 
                    marker='o', label='Pickup Points', zorder=4, alpha=0.8)
    
    # Plot dropoff points
    if total_nodes > max_nodes + 1:
        dropoff_coords = nodes_coordinates[max_nodes+1:2*max_nodes+1]
        plt.scatter(dropoff_coords[:, 0], dropoff_coords[:, 1], s=150, c='crimson', 
                    marker='^', label='Dropoff Points', zorder=4, alpha=0.8)
    
    # Plot path with arrows - consistent color
    path_coords = nodes_coordinates[trajectory]
    total_distance = 0.0
    
    for i in range(len(trajectory) - 1):
        start_coord = path_coords[i]
        end_coord = path_coords[i + 1]
        
        # Calculate distance
        dist = np.sqrt(np.sum((start_coord - end_coord) ** 2))
        total_distance += dist
        
        # Draw arrow with consistent dark blue color
        arrow = FancyArrowPatch(
            start_coord, end_coord,
            arrowstyle='->', 
            mutation_scale=20,
            linewidth=2.5,
            color='cornflowerblue',
            alpha=0.8,
            zorder=3
        )
        plt.gca().add_patch(arrow)
        
        # Add "start" label on the first segment
        if i == 0:
            mid_coord = (start_coord + end_coord) / 2
            plt.annotate('start', mid_coord, xytext=(0, 15), 
                        textcoords='offset points', fontsize=12, 
                        fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    plt.axis('equal')
    
    # Add node labels
    for i, coord in enumerate(nodes_coordinates):
        if i == 0:
            plt.annotate('Depot', (coord[0], coord[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=14, fontweight='bold')
        elif i <= max_nodes:
            plt.annotate(f'P{i}', (coord[0], coord[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=14)
        else:
            plt.annotate(f'D{i-max_nodes}', (coord[0], coord[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=14)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.legend(loc='best', fontsize=14)
    
    plt.text(0.98, 0.02, f'Total Distance: {total_distance:.3f}', 
             transform=plt.gca().transAxes, fontsize=16, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout()
    return total_distance


def make_env(env_id, seed, cfg={}):
    """创建环境"""
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def solve_and_visualize(model_path, max_nodes=10, capacity_limit=20, n_traj=50, device='cuda', seed=42):
    """加载模型并求解CVRP问题"""
    
    # 注册环境
    env_id = 'cvrp-v0'
    env_entry_point = 'envs.cvrp_vector_env:CVRPVectorEnv'
    
    try:
        gym.envs.register(id=env_id, entry_point=env_entry_point)
    except:
        pass
    
    # 创建环境
    envs = SyncVectorEnv([
        make_env(
            env_id, 
            seed, 
            cfg={
                'max_nodes': max_nodes,
                'capacity_limit': capacity_limit,
                'n_traj': n_traj,
                'eval_data': False
            }
        )
    ])
    
    # 创建并加载模型
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    agent = Agent(device=device, name='cvrp').to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"Using device: {device}")
    print("Starting CVRP solving...")
    
    # Solving process
    obs = envs.reset()
    done = np.array([False])
    trajectories = []
    
    while not done.all():
        with torch.no_grad():
            action, logits = agent(obs)
        
        # Multi-greedy inference: first step selects different starting pickup points
        if len(trajectories) == 0:
            pickup_actions = torch.arange(1, max_nodes + 1)
            if n_traj <= max_nodes:
                action = pickup_actions[:n_traj].repeat(1, 1)
            else:
                repeated_actions = pickup_actions.repeat((n_traj // max_nodes) + 1)
                action = repeated_actions[:n_traj].repeat(1, 1)
        
        obs, reward, done, info = envs.step(action.cpu().numpy())
        trajectories.append(action.cpu().numpy())
    
    print(f"Solving completed in {len(trajectories)} steps")
    
    # Get optimal solution
    nodes_coordinates = np.vstack([obs['depot'], obs['observations'][0]])
    final_return = info[0]['episode']['r']
    best_traj = np.argmax(final_return)
    resulting_traj = np.array(trajectories)[:, 0, best_traj]
    resulting_traj_with_depot = np.hstack([np.zeros(1, dtype=int), resulting_traj])
    
    best_distance = -final_return[best_traj]
    
    print(f"Optimal path length: {best_distance:.4f}")
    print(f"Path: {resulting_traj_with_depot}")
    
    caption = f"Ride-sharing CVRP Solution: seed={seed}"
    # Generate visualization
    total_distance = plot_solution(
        nodes_coordinates, 
        resulting_traj_with_depot,
        title=caption
    )
    
    return total_distance


def main():
    parser = argparse.ArgumentParser(description='Ride-sharing CVRP Visualization')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--max-nodes', type=int, default=10, help='Number of pickup points')
    parser.add_argument('--capacity-limit', type=int, default=20, help='Vehicle capacity limit')
    parser.add_argument('--n-traj', type=int, default=50, help='Number of POMO trajectories')
    parser.add_argument('--device', type=str, default='cuda', help='Computing device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='visualization', help='Path to save image')
    
    args = parser.parse_args()
    
    # Solve and visualize
    total_distance = solve_and_visualize(
        args.model_path, 
        args.max_nodes, 
        args.capacity_limit, 
        args.n_traj, 
        args.device, 
        args.seed
    )
    # Save image
    save_path = os.path.join(args.save_dir, f"{args.seed}.png")
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {save_path}")

    # Show image
    plt.show()


if __name__ == "__main__":
    main()