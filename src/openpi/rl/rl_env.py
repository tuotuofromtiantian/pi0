"""
基于 gym_aloha 的强化学习环境
适配 openpi 的 aloha_sim 示例用于 PPO 训练
"""

import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
import logging
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class AlohaRLEnvironment(_environment.Environment):
    """
    基于 gym_aloha 的强化学习环境
    适配用于 PPO 训练
    """

    def __init__(
        self,
        task: str = "gym_aloha/AlohaTransferCube-v0",
        obs_type: str = "pixels_agent_pos",
        seed: int = 0,
        max_steps: int = 200,
        success_threshold: float = 0.05,
        randomize_initial: bool = True,
        reward_config: Optional[Dict[str, float]] = None,
        **kwargs  # 兼容其他参数
    ):
        """
        Args:
            task: gym_aloha 任务名称
            obs_type: 观测类型
            seed: 随机种子
            max_steps: 最大步数
            success_threshold: 成功阈值
            randomize_initial: 是否随机初始化
            reward_config: 奖励配置
        """
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        # 创建 gym 环境
        self._gym = gymnasium.make(task, obs_type=obs_type)

        # 环境参数
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.randomize_initial = randomize_initial

        # 奖励配置
        self.reward_config = reward_config or {
            "distance_weight": 1.0,
            "success_reward": 100.0,
            "action_penalty": 0.1,
            "time_penalty": 0.01,
            "failure_penalty": -50.0,
        }

        # 状态追踪
        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0
        self._current_step = 0
        self._last_cube_pos = None
        self._target_pos = None
        self._prev_cube_pos = None
        self._step_reward = 0.0

        # 获取动作和观测空间信息
        self._setup_spaces()

    def _setup_spaces(self):
        """设置动作和观测空间"""
        # 获取原始空间
        gym_action_space = self._gym.action_space
        gym_obs_space = self._gym.observation_space

        # 动作空间（通常是7维或14维连续动作）
        if isinstance(gym_action_space, spaces.Box):
            self.action_dim = gym_action_space.shape[0]
        else:
            self.action_dim = 7  # 默认7维

        logging.info(f"Action space dimension: {self.action_dim}")

    @override
    def reset(self) -> None:
        """重置环境"""
        # 重置 gym 环境
        gym_obs, info = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))

        # 随机化初始条件（如果需要）
        if self.randomize_initial and hasattr(self._gym, 'sim'):
            self._randomize_initial_state()

        # 转换观测
        self._last_obs = self._convert_observation(gym_obs)

        # 重置状态
        self._done = False
        self._episode_reward = 0.0
        self._current_step = 0
        self._step_reward = 0.0

        # 提取初始位置信息
        self._extract_positions(gym_obs)
        self._prev_cube_pos = None

    def _randomize_initial_state(self):
        """随机化初始状态"""
        # 这里可以添加对 MuJoCo 仿真状态的随机化
        # 例如：随机化物体位置、机器人关节角度等
        pass

    def _extract_positions(self, gym_obs):
        """从观测中提取位置信息"""
        # 根据具体的 gym_aloha 环境，提取相关位置
        # 这里需要根据实际环境结构调整
        if isinstance(gym_obs, dict):
            if 'object_pos' in gym_obs:
                self._last_cube_pos = gym_obs['object_pos'].copy()
            elif 'observation' in gym_obs and len(gym_obs['observation']) >= 3:
                # 假设前3维是物体位置
                self._last_cube_pos = gym_obs['observation'][:3].copy()

            # 设置目标位置（根据任务不同可能需要调整）
            if 'goal_pos' in gym_obs:
                self._target_pos = gym_obs['goal_pos'].copy()
            else:
                # 默认目标位置
                self._target_pos = np.array([0.5, 0.0, 0.1])

    @override
    def is_episode_complete(self) -> bool:
        """检查episode是否结束"""
        return self._done

    @override
    def get_observation(self) -> dict:
        """获取当前观测"""
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")
        return self._last_obs

    @override
    def apply_action(self, action: dict) -> None:
        """应用动作"""
        # 提取动作数组
        if isinstance(action, dict):
            action_array = action.get("actions", action.get("action"))
        else:
            action_array = action

        # 执行动作
        gym_obs, gym_reward, terminated, truncated, info = self._gym.step(action_array)

        # 更新步数
        self._current_step += 1

        # 转换观测
        self._last_obs = self._convert_observation(gym_obs)

        # 提取位置信息
        self._extract_positions(gym_obs)

        # 计算自定义奖励
        self._step_reward = self._compute_reward(gym_obs, action_array, gym_reward)
        self._episode_reward += self._step_reward

        # 检查终止条件
        self._done = terminated or truncated or self._current_step >= self.max_steps

        # 添加成功信息
        self._is_success = self._check_success(gym_obs)

    def _convert_observation(self, gym_obs: dict) -> dict:
        """转换观测格式"""
        obs = {}

        # 处理图像
        if isinstance(gym_obs, dict) and "pixels" in gym_obs:
            img = gym_obs["pixels"]["top"]
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
            # 转换轴顺序 [H, W, C] -> [C, H, W]
            img = np.transpose(img, (2, 0, 1))
            obs["images"] = {"cam_high": img}

        # 处理状态向量
        if isinstance(gym_obs, dict) and "agent_pos" in gym_obs:
            # 使用原始的 agent_pos
            state = gym_obs["agent_pos"]
        elif isinstance(gym_obs, np.ndarray):
            # 如果是纯数组，直接使用
            state = gym_obs
        else:
            # 尝试从其他字段构建状态
            state = self._build_state_vector(gym_obs)

        # 确保状态是32维（与 Pi0 兼容）
        if len(state) < 32:
            state = np.pad(state, (0, 32 - len(state)), mode='constant')
        elif len(state) > 32:
            state = state[:32]

        obs["state"] = state.astype(np.float32)

        return obs

    def _build_state_vector(self, gym_obs):
        """构建状态向量"""
        state_parts = []

        # 根据 gym_obs 的结构构建状态
        if isinstance(gym_obs, dict):
            # 添加各种可能的状态组件
            for key in ['agent_pos', 'object_pos', 'goal_pos', 'observation']:
                if key in gym_obs:
                    data = gym_obs[key]
                    if isinstance(data, np.ndarray):
                        state_parts.append(data.flatten())

        if state_parts:
            return np.concatenate(state_parts)
        else:
            # 返回默认状态
            return np.zeros(32)

    def _compute_reward(self, gym_obs, action, gym_reward):
        """计算自定义奖励"""
        config = self.reward_config
        total_reward = 0.0

        # 1. 使用原始 gym 奖励作为基础
        total_reward += gym_reward

        # 2. 动作惩罚
        action_penalty = -config["action_penalty"] * np.linalg.norm(action)
        total_reward += action_penalty

        # 3. 时间惩罚
        time_penalty = -config["time_penalty"]
        total_reward += time_penalty

        # 4. 距离奖励（如果有位置信息）
        if self._last_cube_pos is not None and self._target_pos is not None:
            dist = np.linalg.norm(self._last_cube_pos - self._target_pos)
            distance_reward = -dist * config["distance_weight"]
            total_reward += distance_reward

            # 5. 成功奖励
            if dist < self.success_threshold:
                total_reward += config["success_reward"]

        # 6. 进步奖励（如果有历史位置）
        if self._prev_cube_pos is not None:
            if self._last_cube_pos is not None and self._target_pos is not None:
                prev_dist = np.linalg.norm(self._prev_cube_pos - self._target_pos)
                curr_dist = np.linalg.norm(self._last_cube_pos - self._target_pos)
                progress = prev_dist - curr_dist
                progress_reward = progress * 10.0
                total_reward += progress_reward

        # 更新历史位置
        if self._last_cube_pos is not None:
            self._prev_cube_pos = self._last_cube_pos.copy()

        return total_reward

    def _check_success(self, gym_obs):
        """检查是否成功"""
        if self._last_cube_pos is not None and self._target_pos is not None:
            dist = np.linalg.norm(self._last_cube_pos - self._target_pos)
            return dist < self.success_threshold
        return False

    def get_info(self) -> dict:
        """获取额外信息"""
        info = {
            "is_success": getattr(self, "_is_success", False),
            "episode_reward": self._episode_reward,
            "current_step": self._current_step,
            "step_reward": self._step_reward,
        }

        if self._last_cube_pos is not None and self._target_pos is not None:
            dist = np.linalg.norm(self._last_cube_pos - self._target_pos)
            info["distance_to_goal"] = dist

        return info


class AlohaRLGymWrapper(gymnasium.Env):
    """
    将 AlohaRLEnvironment 包装为标准 Gym 接口
    用于 stable-baselines3
    """

    def __init__(self, env_config: dict):
        super().__init__()

        # 创建内部环境
        self.env = AlohaRLEnvironment(**env_config)

        # 设置动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.env.action_dim,),
            dtype=np.float32
        )

        # 设置观测空间
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(32,),
                dtype=np.float32
            ),
        })

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        self.env.reset()
        obs = self.env.get_observation()
        info = self.env.get_info()

        return obs, info

    def step(self, action):
        """执行一步"""
        self.env.apply_action({"actions": action})

        obs = self.env.get_observation()
        info = self.env.get_info()

        # 获取步奖励而不是累积奖励
        reward = info.get("step_reward", 0.0)
        done = self.env.is_episode_complete()
        truncated = False

        return obs, reward, done, truncated, info

    def render(self):
        """渲染（可选）"""
        pass

    def close(self):
        """关闭环境"""
        if hasattr(self.env, '_gym'):
            self.env._gym.close()


# 兼容性别名 - 保持向后兼容
MoveObjectEnv = AlohaRLEnvironment
MoveObjectGymWrapper = AlohaRLGymWrapper