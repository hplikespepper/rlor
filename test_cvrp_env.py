# test_cvrp_env.py - 单独测试CVRP环境的脚本
import gym
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append('.')

# 注册环境
gym.envs.register(
    id='cvrp-v0',
    entry_point='envs.cvrp_vector_env:CVRPVectorEnv',
)

def test_cvrp_environment():
    """测试CVRP环境是否能正常工作"""
    print("=== CVRP Environment Test ===")
    
    # 创建环境
    env_config = {
        'max_nodes': 2,
        'capacity_limit': 20,
        'n_traj': 4,  # 使用较少的轨迹便于调试
        'eval_data': False,
        'debug': True,
    }
    
    env = gym.make('cvrp-v0', **env_config)
    print(f"Environment created successfully")
    print(f"Action space: {env.action_space}")
    print(f"Total nodes: {env.total_nodes}")
    print(f"Expected episode length: ~{2 * env.max_nodes + 5} steps")
    
    # 重置环境
    obs = env.reset()
    print(f"\nInitial observation keys: {list(obs.keys())}")
    print(f"Action mask shape: {obs['action_mask'].shape}")
    
    # 手动执行一个完整的episode
    print("\n=== Manual Episode Execution ===")
    step = 0
    max_steps = 50
    episode_completed = False
    
    while step < max_steps and not episode_completed:
        print(f"\n--- Step {step + 1} ---")
        
        # 显示当前状态
        print(f"Current positions (last_node_idx): {obs['last_node_idx']}")
        print(f"Current loads: {obs['current_load']}")
        
        # 为每个轨迹选择第一个有效动作
        actions = []
        for traj in range(env.n_traj):
            valid_actions = np.where(obs['action_mask'][traj])[0]
            if len(valid_actions) > 0:
                # 智能选择策略：
                # 1. 如果在depot且有未完成任务，优先选择取货点
                # 2. 如果有货物，优先选择对应的送货点
                # 3. 否则选择第一个有效动作
                
                current_pos = obs['last_node_idx'][traj]
                current_load = obs['current_load'][traj]
                
                if current_pos == 0 and current_load == 0:  # 在depot且空载
                    # 优先选择取货点 (1 到 max_nodes)
                    pickup_actions = [a for a in valid_actions if 1 <= a <= env.max_nodes]
                    if pickup_actions:
                        action = pickup_actions[0]
                    else:
                        action = valid_actions[0]
                elif current_load > 0:  # 有货物，寻找送货点
                    # 寻找对应的送货点
                    delivery_actions = [a for a in valid_actions if a > env.max_nodes]
                    if delivery_actions:
                        action = delivery_actions[0]
                    else:
                        action = valid_actions[0]
                else:
                    action = valid_actions[0]
                
                actions.append(action)
                print(f"  Traj {traj}: pos={current_pos}, load={current_load:.1f}, "
                      f"valid_actions={valid_actions}, chosen={action}")
            else:
                # 没有有效动作，这是个问题
                actions.append(0)  # 默认回depot
                print(f"  Traj {traj}: NO VALID ACTIONS! Defaulting to depot (0)")
        
        actions = np.array(actions)
        
        # 执行动作
        obs, rewards, dones, info = env.step(actions)
        
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")
        
        # 检查是否有episode完成
        completed_count = dones.sum()
        if completed_count > 0:
            print(f"*** {completed_count} episodes completed at step {step + 1}! ***")
            episode_completed = True
        
        step += 1
    
    if not episode_completed:
        print(f"\nNo episodes completed after {max_steps} steps")
        print("This indicates a problem with the environment logic")
    else:
        print(f"\nSuccess! Episodes completed in {step} steps")
    
    env.close()
    return episode_completed

def test_simple_policy():
    """测试一个简单的策略能否完成任务"""
    print("\n=== Simple Policy Test ===")
    
    env_config = {
        'max_nodes': 2,
        'capacity_limit': 20,
        'n_traj': 1,  # 只用一个轨迹
        'eval_data': False,
        'debug': True,
    }
    
    env = gym.make('cvrp-v0', **env_config)
    obs = env.reset()
    
    # 预定义的动作序列：取货1 -> 送货1 -> 取货2 -> 送货2 -> 回depot
    # 对于max_nodes=2: 取货点是1,2；送货点是3,4；depot是0
    planned_actions = [1, 3, 2, 4, 0]  # 取货1, 送货1, 取货2, 送货2, 回depot
    
    print(f"Planned action sequence: {planned_actions}")
    print("Expected: pickup1 -> delivery1 -> pickup2 -> delivery2 -> depot")
    
    for step, planned_action in enumerate(planned_actions):
        print(f"\nStep {step + 1}: Planned action = {planned_action}")
        
        # 检查这个动作是否有效
        if obs['action_mask'][0, planned_action]:
            action = np.array([planned_action])
            print(f"  Action is valid, executing...")
        else:
            # 如果计划动作无效，选择第一个有效动作
            valid_actions = np.where(obs['action_mask'][0])[0]
            action = np.array([valid_actions[0]])
            print(f"  Planned action invalid, using {action[0]} instead")
            print(f"  Valid actions were: {valid_actions}")
        
        obs, reward, done, info = env.step(action)
        
        print(f"  Current position: {obs['last_node_idx'][0]}")
        print(f"  Current load: {obs['current_load'][0]}")
        print(f"  Reward: {reward[0]:.3f}")
        print(f"  Done: {done[0]}")
        
        if done[0]:
            print(f"*** Episode completed at step {step + 1}! ***")
            break
    
    env.close()

if __name__ == "__main__":
    print("Testing CVRP Environment...")
    
    try:
        # 测试1：基本环境功能
        success = test_cvrp_environment()
        
        # 测试2：简单策略
        test_simple_policy()
        
        if success:
            print("\n✅ Environment test PASSED")
        else:
            print("\n❌ Environment test FAILED")
            
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        import traceback
        traceback.print_exc()