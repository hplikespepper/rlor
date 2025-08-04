# import numpy as np
# from envs.cvrp_vector_env import CVRPVectorEnv

# def test_env():
#     # 1）创建一个只有 1 条轨迹的小环境，3 个取货点，容量上限 2
#     env = CVRPVectorEnv(max_nodes=3, capacity_limit=2, n_traj=1)
#     obs = env.reset()
#     print("=== Reset 后 ===")
#     print("current_load:", obs["current_load"])       # 期望 [0.0]
#     print("action_mask:", obs["action_mask"][0])     # 期望 depot False，1,2,3 True，其余 False

#     # 2）访问取货点 1
#     action = np.array([1], dtype=int)
#     obs, rew, done, info = env.step(action)
#     print("\n=== Step: 访问取货点 1 ===")
#     print("current_load:", obs["current_load"])       # 期望 [1.0]
#     print("mask:", obs["action_mask"][0])
#     print("done:", done)                              # 期望 [False]

#     # 3）访问取货点 2
#     action = np.array([2], dtype=int)
#     obs, rew, done, info = env.step(action)
#     print("\n=== Step: 访问取货点 2 ===")
#     print("current_load:", obs["current_load"])       # 期望 [2.0]
#     print("mask:", obs["action_mask"][0])
#     print("done:", done)                              # 期望 [False]

#     # 4）访问送货点 4（对应第1个取客）
#     action = np.array([4], dtype=int)
#     obs, rew, done, info = env.step(action)
#     print("\n=== Step: 访问送货点 4 ===")
#     print("current_load:", obs["current_load"])       # 期望 [1.0]
#     print("mask:", obs["action_mask"][0])
#     print("done:", done)                              # 期望 [False]

#     # 5）访问送货点 5（对应第2个取客）
#     action = np.array([5], dtype=int)
#     obs, rew, done, info = env.step(action)
#     print("\n=== Step: 访问送货点 5 ===")
#     print("current_load:", obs["current_load"])       # 期望 [0.0]
#     print("mask:", obs["action_mask"][0])
#     print("done after last dropoff:", done)           # 期望 [True]

#     # 6）（可选）访问 depot，观察环境是否会重置
#     action = np.array([0], dtype=int)
#     obs, rew, done, info = env.step(action)
#     print("\n=== Step: 返回 depot ===")
#     print("current_load:", obs["current_load"])       # 期望 [0.0]
#     print("mask:", obs["action_mask"][0])
#     print("done after depot:", done)                  # 期望 [True]

# if __name__ == "__main__":
#     test_env()

import numpy as np
from envs.cvrp_vector_env import CVRPVectorEnv

def test_env():
    env = CVRPVectorEnv(max_nodes=3, capacity_limit=2, n_traj=1)
    obs = env.reset()

    # 依次服务 3 位乘客：取货 1→送货 4→取货 2→送货 5→取货 3→送货 6
    sequence = [1, 4, 2, 5, 3, 6]
    for step_idx, a in enumerate(sequence, 1):
        obs, rew, done, info = env.step(np.array([a]))
        print(f"Step {step_idx}: action={a}")
        print("  current_load:", obs["current_load"])
        print("  mask:", obs["action_mask"][0])
        print("  done:", done)
        print("-" * 40)
    
    # 此时所有节点都已访问，done 应该为 True
    # 你也可以再试一次返回 depot，看 done 是否依旧 True
    obs, rew, done, info = env.step(np.array([0]))
    print("After full service, step to depot:")
    print("  done:", done)

if __name__ == "__main__":
    test_env()

