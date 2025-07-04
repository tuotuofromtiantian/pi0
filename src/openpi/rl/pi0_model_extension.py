"""
Pi0 模型的扩展功能
为强化学习添加必要的接口
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, Any
from openpi.models.pi0 import Pi0
from openpi.models import model as _model
from openpi.shared import array_typing as at
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Union
import torch                # ← 解决 NameError
import torch.nn as nn
import numpy as np
import jax.numpy as jnp
from gymnasium import spaces
class Pi0WithRLFeatures(Pi0):
    """
    扩展的 Pi0 模型，添加强化学习所需的特征提取接口
    """

    def __init__(self, config, rngs):
        super().__init__(config, rngs)

        # 添加一个特征维度属性
        self._feature_dim = None

    @property
    def feature_dim(self) -> int:
        """获取特征维度"""
        if self._feature_dim is None:
            # 通过运行一个虚拟输入来推断特征维度
            dummy_obs = self._create_dummy_observation()
            features = self.extract_features(dummy_obs)
            self._feature_dim = features.shape[-1]
        return self._feature_dim

    def _create_dummy_observation(self) -> _model.Observation:
        """创建用于推断维度的虚拟观测"""
        batch_size = 1
        dummy_image = jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32)
        dummy_image_mask = jnp.ones((batch_size,), dtype=jnp.bool_)

        return _model.Observation(
            images={"cam_high": dummy_image},
            image_masks={"cam_high": dummy_image_mask},
            state=jnp.zeros((batch_size, self.action_dim), dtype=jnp.float32),
            tokenized_prompt=jnp.zeros((batch_size, self.max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.zeros((batch_size, self.max_token_len), dtype=jnp.bool_),
        )

    @at.typecheck
    def extract_features(
            self,
            observation: _model.Observation,
            return_separate: bool = False
    ) -> Union[at.Float[at.Array, "b d"], Tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b d"]]]:
        """
        提取用于策略和价值函数的特征

        Args:
            observation: 输入观测
            return_separate: 是否分别返回 actor 和 critic 特征

        Returns:
            features: 特征向量，或 (actor_features, critic_features) 元组
        """
        # 获取前缀嵌入（图像和语言）
        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)

        # 聚合特征的几种方式：
        # 1. 平均池化所有有效 token
        valid_tokens = prefix_tokens * prefix_mask[:, :, None]
        num_valid = jnp.sum(prefix_mask, axis=1, keepdims=True)
        avg_features = jnp.sum(valid_tokens, axis=1) / jnp.maximum(num_valid, 1)

        # 2. 使用 CLS token（如果存在）或第一个 token
        # first_token_features = prefix_tokens[:, 0, :]

        # 3. 最大池化
        # max_features = jnp.max(valid_tokens, axis=1)

        # 这里我们使用平均池化作为默认
        features = avg_features

        if return_separate:
            # 可以为 actor 和 critic 使用不同的聚合方式
            # 或者添加额外的投影层
            actor_features = features
            critic_features = features  # 或使用不同的处理
            return actor_features, critic_features
        else:
            return features

    @at.typecheck
    def get_action_distribution_params(
            self,
            observation: _model.Observation,
            features: Optional[at.Float[at.Array, "b d"]] = None
    ) -> Tuple[at.Float[at.Array, "b a"], at.Float[at.Array, "b a"]]:
        """
        获取动作分布的参数（均值和标准差）

        Args:
            observation: 输入观测
            features: 预计算的特征（可选）

        Returns:
            mean: 动作均值
            log_std: 动作对数标准差
        """
        if features is None:
            features = self.extract_features(observation)

        # 通过动作投影层获取均值
        # 注意：这里需要根据实际的 Pi0 结构调整
        # 假设我们有一个从特征到动作的投影
        mean = self.action_out_proj(features)

        # 对于标准差，我们可以：
        # 1. 使用固定的标准差
        # 2. 学习一个与状态相关的标准差
        # 3. 使用可学习的参数

        # 这里使用简单的固定标准差
        log_std = jnp.full_like(mean, -1.0)  # std ≈ 0.37

        return mean, log_std

    @at.typecheck
    def compute_value(
            self,
            observation: _model.Observation,
            value_head: nnx.Module
    ) -> at.Float[at.Array, "b 1"]:
        """
        计算状态价值（需要外部提供价值头）

        Args:
            observation: 输入观测
            value_head: 价值函数头（外部提供）

        Returns:
            value: 状态价值估计
        """
        features = self.extract_features(observation)
        value = value_head(features)
        return value

    @at.typecheck
    def get_policy_regularization_loss(
            self,
            observation: _model.Observation,
            actions: _model.Actions,
            old_log_probs: Optional[at.Float[at.Array, "b a"]] = None
    ) -> at.Float[at.Array, ""]:
        """
        计算策略正则化损失（如 KL 散度）

        Args:
            observation: 输入观测
            actions: 采取的动作
            old_log_probs: 旧策略的对数概率（用于 PPO）

        Returns:
            reg_loss: 正则化损失
        """
        # 获取当前策略参数
        mean, log_std = self.get_action_distribution_params(observation)
        std = jnp.exp(log_std)

        # 计算当前动作的对数概率
        log_probs = -0.5 * jnp.sum(
            jnp.square((actions - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1
        )

        if old_log_probs is not None:
            # PPO 风格的 KL 散度近似
            kl_div = old_log_probs - log_probs
            return jnp.mean(kl_div)
        else:
            # 或者使用熵正则化
            entropy = 0.5 * jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
            return -jnp.mean(entropy)  # 负熵作为正则化


def add_rl_features_to_pi0(pi0_model: Pi0) -> Pi0WithRLFeatures:
    """
    将现有的 Pi0 模型转换为带有 rl 特征的版本

    Args:
        pi0_model: 原始 Pi0 模型

    Returns:
        pi0_with_features: 扩展后的模型
    """
    # 创建新的模型实例
    config = pi0_model.config if hasattr(pi0_model, 'config') else None

    # 复制权重
    # 注意：这里需要根据实际的 nnx 实现来处理权重复制
    # 简化示例：
    extended_model = Pi0WithRLFeatures(config, nnx.Rngs(0))

    # 复制所有参数
    # 这需要遍历模型的所有参数并复制
    # 具体实现取决于 nnx 的 API

    return extended_model


# 用于 PyTorch 和 JAX 之间转换的辅助函数
def jax_to_torch(x: jnp.ndarray) -> torch.Tensor:
    """将 JAX 数组转换为 PyTorch tensor"""
    import torch
    return torch.from_numpy(np.asarray(x))


def torch_to_jax(x: torch.Tensor) -> jnp.ndarray:
    """将 PyTorch tensor 转换为 JAX 数组"""
    return jnp.asarray(x.detach().cpu().numpy())


class Pi0TorchWrapper(torch.nn.Module):
    """
    将 Pi0 模型包装为 PyTorch 模块
    用于在 PyTorch 框架中使用
    """

    def __init__(self, pi0_model: Pi0WithRLFeatures):
        super().__init__()
        self.pi0_model = pi0_model

        # JIT 编译关键函数以提高性能
        self._extract_features_jit = jax.jit(self.pi0_model.extract_features)
        self._get_action_params_jit = jax.jit(self.pi0_model.get_action_distribution_params)

        # 创建虚拟参数以满足 PyTorch 的要求
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(
            self,
            obs_dict: Dict[str, torch.Tensor],
            return_features: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        前向传播

        Args:
            obs_dict: 观测字典（PyTorch tensors）
            return_features: 是否返回特征

        Returns:
            如果 return_features=True: 特征向量
            否则: (action_mean, action_log_std) 元组
        """
        # 转换为 JAX 格式
        obs_jax = {k: torch_to_jax(v) for k, v in obs_dict.items()}
        observation = _model.Observation.from_dict(obs_jax)

        if return_features:
            # 提取特征
            features_jax = self._extract_features_jit(observation)
            features_torch = jax_to_torch(features_jax)
            return features_torch
        else:
            # 获取动作分布参数
            mean_jax, log_std_jax = self._get_action_params_jit(observation)
            mean_torch = jax_to_torch(mean_jax)
            log_std_torch = jax_to_torch(log_std_jax)
            return mean_torch, log_std_torch

    def extract_features(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取特征的便捷方法"""
        return self.forward(obs_dict, return_features=True)

    def get_action_dist_params(
            self,
            obs_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作分布参数的便捷方法"""
        return self.forward(obs_dict, return_features=False)

class Pi0FeatureExtractor(nn.Module):
    """
    Stable-Baselines3 自定义特征抽取器：
      · 先把 SB3 的 torch 观测 Dict → numpy → JAX
      · 调用 Pi-0 (JAX/Flax) 提特征
      · 再把特征转回 torch，并做一次线性投影到 features_dim
    NOTE:
      · 必须继承 nn.Module 而不是 BaseFeaturesExtractor，
        因为我们自己手动存储 observation_space / features_dim 即可。
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        pi0_model: Pi0WithRLFeatures,
        transforms,
        features_dim: int = 256,
    ):
        super().__init__()
        # ① 先存 observation_space，供后续 _create_dummy_observation 使用
        self.observation_space = observation_space

        # ② 暴露给上层（load_pi0_model 里要用到）
        self.pi0_model = pi0_model
        self.transforms = transforms

        # ③ 推断 Pi-0 输出维度（跑一次 dummy 前向即可）
        with jax.disable_jit():                 # 避免第一次 compile 很慢
            dummy_obs = self._create_dummy_observation()
            jax_feat  = self.pi0_model.extract_features(dummy_obs)
            self.pi0_feature_dim = int(jax_feat.shape[-1])

        # ④ 把 JAX 特征映射到 SB3 期望的 features_dim
        self.proj = nn.Linear(self.pi0_feature_dim, features_dim)

    # ======= 工具函数 =======
    def _create_dummy_observation(self) -> _model.Observation:
        """
        根据 observation_space 构造 **最小** 可用 dummy obs
        这里只演示常见三个键：'rgb', 'state', 'prompt'（请按需增删）
        """
        batch = 1
        # 1) 图像
        rgb_shape = self.observation_space["rgb"].shape   # (H,W,3)
        rgb_dummy = np.zeros((batch, *rgb_shape), dtype=np.float32)

        # 2) 机器人状态（连续向量）
        state_shape = self.observation_space["state"].shape
        state_dummy = np.zeros((batch, *state_shape), dtype=np.float32)

        # 3) Prompt（token IDs）
        prompt_len  = self.observation_space["prompt"].shape[0]
        prompt_dummy = np.zeros((batch, prompt_len), dtype=np.int32)

        return _model.Observation.from_dict(
            {
                "images"             : {"cam_high": jnp.asarray(rgb_dummy)},
                "image_masks"        : {"cam_high": jnp.ones((batch,), dtype=jnp.bool_)},
                "state"              : jnp.asarray(state_dummy),
                "tokenized_prompt"   : jnp.asarray(prompt_dummy),
                "tokenized_prompt_mask": jnp.zeros_like(prompt_dummy, dtype=jnp.bool_),
            }
        )

    @staticmethod
    def _torchify(jax_array: jnp.ndarray, device: torch.device) -> torch.Tensor:
        """JAX → torch（保持 float32）"""
        return torch.as_tensor(np.asarray(jax_array), device=device, dtype=torch.float32)
    # =======================

    # -------- 核心前向 --------
    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        SB3 会把 Gym Dict 打包成 {key: torch.Tensor}
        我们要先把它们同步到 CPU，然后转 numpy → JAX
        """
        # 1) torch → numpy → JAX
        obs_np = {k: v.detach().cpu().numpy() for k, v in observations.items()}
        jax_obs = _model.Observation.from_dict(obs_np)

        # 2) 提特征（JAX）
        jax_feat = self.pi0_model.extract_features(jax_obs)

        # 3) JAX → torch，并投影到 features_dim
        device = next(self.parameters()).device
        torch_feat = self._torchify(jax_feat, device)
        return self.proj(torch_feat)