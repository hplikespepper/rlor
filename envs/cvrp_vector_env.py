# 文件路径：envs/cvrp_vector_env.py

import gym
import numpy as np
from gym import spaces

from .vrp_data import VRPDataset

def assign_env_config(self, kwargs):
    """
    将 kwargs 中的 key, value 赋给 self
    """
    for key, value in kwargs.items():
        setattr(self, key, value)

def dist(loc1, loc2):
    return ((loc1[:, 0] - loc2[:, 0])**2 + (loc1[:, 1] - loc2[:, 1])**2)**0.5

class CVRPVectorEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        # 原始配置
        self.max_nodes = 20           # N，原始客户数量
        self.capacity_limit = 40      # 车辆最大载客量
        self.n_traj = 50              # POMO 轨迹数
        self.eval_data = False
        self.eval_partition = "test"
        self.eval_data_idx = 0
        self.demand_limit = 10        # 随机生成时的最大需求
        # 覆盖
        assign_env_config(self, kwargs)

        # 总节点数 = Depot + N 取货点 + N 送货点
        self.total_nodes = 2 * self.max_nodes + 1

        # 定义 Observation Space
        obs_dict = {
            # 除了 depot，其它节点均在 observations 中 (shape=(total_nodes-1,2))
            "observations": spaces.Box(low=0, high=1, shape=(self.total_nodes - 1, 2)),
            "depot": spaces.Box(low=0, high=1, shape=(2,)),
            # demand: 取货点为 +1，送货点为 -1，对应 normalize 到 [0,1]
            "demand": spaces.Box(low=-1, high=1, shape=(self.total_nodes - 1,)),
            # 掩码：只能访问未访问且合法的节点
            "action_mask": spaces.MultiBinary([self.n_traj, self.total_nodes]),
            "last_node_idx": spaces.MultiDiscrete([self.total_nodes] * self.n_traj),
            "current_load": spaces.Box(low=0, high=self.capacity_limit, shape=(self.n_traj,)),
            "capacity_limit": spaces.Box(low=0, high=self.capacity_limit, shape=(self.n_traj,)),
        }
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.total_nodes] * self.n_traj)
        self.reward_space = None

        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def cost(self, loc1, loc2):
        return dist(loc1, loc2)

    def is_all_visited(self):
        # 当所有取货和送货点均已访问过一次
        return self.visited[:, 1:].all(axis=1)

    # def _update_mask(self):
    #     # 1: 未访问; 0: 已访问
    #     mask = ~self.visited

    #     # # Depot (索引 0) 仅当已完成所有访问或上一步不在 Depot 时可选
    #     # mask[:, 0] |= (self.last != 0) | self.is_all_visited()

    #     # depot 只能在所有取送完成后才可访问
    #     all_served = self.is_all_visited()
    #     mask[:, 0] &= all_served

    #     # 对于送货点（索引 > N 且 <= 2N），只有在对应取货点已访问后才可选
    #     # 假设取货点索引范围 [1, N]，送货点索引范围 [N+1, 2N]
    #     # 取货点 i 对应送货点 i+N
    #     for i in range(1, self.max_nodes + 1):
    #         # 如果取货点 i 未访问，则屏蔽送货点 i+self.max_nodes
    #         mask[:, i + self.max_nodes] &= self.visited[:, i]

    #     # 取货点只能在当前载客量 < capacity_limit 时访问
    #     mask[:, 1 : self.max_nodes + 1] &= (self.load[:, None] < self.capacity_limit)

    #     return mask

    def _update_mask(self):
        mask = ~self.visited
        mask[:, 0] = True  # 总是允许回depot
        mask[:, 1:self.max_nodes + 1] &= (self.load[:, None] < self.capacity_limit)
        
        for i in range(1, self.max_nodes + 1):
            delivery_idx = i + self.max_nodes
            if delivery_idx < self.total_nodes:
                mask[:, delivery_idx] &= self.visited[:, i]
        
        # 确保至少有一个有效动作
        no_valid_actions = ~mask.any(axis=1)
        if no_valid_actions.any():
            mask[no_valid_actions, 0] = True
        return mask

    def _update_state(self):
        obs = {
            "observations": self.nodes[1:],  # 除 Depot 外的节点
            "depot": self.nodes[0],
            "action_mask": self._update_mask(),
            "demand": self.demands,
            "last_node_idx": self.last,
            "current_load": self.load,
            "capacity_limit": np.full((self.n_traj,), self.capacity_limit, dtype=np.float32)
        }
        return obs

    def _go_to(self, destin):
        # destin: shape (n_traj,), 每条轨迹要访问的节点索引
        dest_coord = self.nodes[destin]
        d = self.cost(dest_coord, self.nodes[self.last])
        self.last = destin

        # 访问 Depot：重置载客量
        self.load[destin == 0] = 0.0

        # 访问取货点 1..N：载客量 +1
        pickup_mask = (destin > 0) & (destin <= self.max_nodes)
        self.load[pickup_mask] += 1.0
        # 访问送货点 N+1..2N：载客量 -1
        dropoff_mask = destin > self.max_nodes
        self.load[dropoff_mask] -= 1.0

        # 将该节点标记为已访问
        self.visited[np.arange(self.n_traj), destin] = True

        # 奖励为负距离
        self.reward = -d

    def _LOAD_OR_GENERATE(self):
        if self.eval_data:
            # 从数据集中读取（暂不支持 pre-generated Pickup-Delivery 数据，建议只用随机生成做训练/测试）
            data = VRPDataset[self.eval_partition, self.max_nodes, self.eval_data_idx]
            self.nodes = np.concatenate((data["depot"][None, ...], data["loc"]))
            # 原始只含 N 个 demand；此处不建议用 eval_data
            self.demands = data["demand"]
        else:
            # 随机生成 Depot + N 取货 + N 送货
            depot = np.random.rand(1, 2)
            pickups = np.random.rand(self.max_nodes, 2)
            dropoffs = np.random.rand(self.max_nodes, 2)
            self.nodes = np.vstack((depot, pickups, dropoffs))  # shape=(2N+1,2)
            # demand: 取货点 +1，送货点 -1
            self.demands = np.concatenate(
                (np.ones(self.max_nodes), -np.ones(self.max_nodes))
            )

        # 初始化为未访问
        self.visited = np.zeros((self.n_traj, self.total_nodes), dtype=bool)
        # Depot 默认已访问
        # 关键修复：不要标记depot为已访问
        # self.visited[:, 0] = True
        # 最后访问节点均为 Depot
        self.last = np.zeros(self.n_traj, dtype=int)
        # 载客量归 0
        self.load = np.zeros(self.n_traj, dtype=float)

    def reset(self):
        self.num_steps = 0
        self._LOAD_OR_GENERATE()
        self.state = self._update_state()
        self.done = np.zeros(self.n_traj, dtype=bool)
        self.info = {}
        return self.state

    def step(self, action):
        """
        action: array of shape (n_traj,), 每个元素为 [0, 2N] 之间的整数
        """
        self._go_to(action)               # 更新位置、载客量、奖励
        self.num_steps += 1
        self.state = self._update_state()
        # 完成条件：返回 Depot 且已访问所有节点
        self.done = (action == 0) & self.is_all_visited()
        # self.done = self.is_all_visited()


        return self.state, self.reward, self.done, self.info
