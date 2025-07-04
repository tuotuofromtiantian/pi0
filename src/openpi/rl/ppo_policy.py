"""
自定义 Actor-Critic 策略，集成 Pi0 预训练模型
基于 stable-baselines3 框架
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Type, Union, Optional, Any
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.type_aliases import Schedule

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi import transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.rl.pi0_model_extension import Pi0FeatureExtractor
from typing import Union


class Pi0FeatureExtractor(BaseFeaturesExtractor):
    """
    使用 Pi0 模型作为特征提取器
    将 JAX/Flax 模型集成到 PyTorch 框架中
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            pi0_model: Pi0,
            features_dim: int = 256,
            transforms: List[_transforms.DataTransformFn] = None,
            freeze_pi0: bool = False,
    ):
        """
        Args:
            observation_space: 观测空间
            pi0_model: 预训练的 Pi0 模型
            features_dim: 输出特征维度
            transforms: 数据预处理转换
            freeze_pi0: 是否冻结 Pi0 参数
        """
        super().__init__(observation_space, features_dim)

        self.pi0_model = pi0_model
        self.transforms = _transforms.compose(transforms or [])
        self.freeze_pi0 = freeze_pi0

        # 创建特征投影层（从 Pi0 特征到统一维度）
        # 注意：这里需要知道 Pi0 的实际输出维度
        pi0_feature_dim = self._get_pi0_feature_dim()
        self.feature_proj = nn.Linear(pi0_feature_dim, features_dim)

        # 如果冻结 Pi0，创建一个 JIT 编译的推理函数
        if freeze_pi0:
            self._pi0_extract_features = self._create_jit_feature_extractor()

    def _get_pi0_feature_dim(self) -> int:
        """获取 Pi0 模型的特征维度"""
        # 这需要根据实际的 Pi0 实现来确定
        # 假设 Pi0 模型有一个方法返回特征维度
        # 或者通过运行一个样本来推断
        dummy_obs = self._create_dummy_observation()
        with torch.no_grad():
            features = self._extract_pi0_features(dummy_obs)
        return features.shape[-1]

    def _create_dummy_observation(self) -> Dict[str, Any]:
        """创建用于推断维度的虚拟观测"""
        if isinstance(self.observation_space, spaces.Dict):
            dummy_obs = {}
            for key, space in self.observation_space.spaces.items():
                if isinstance(space, spaces.Box):
                    dummy_obs[key] = np.zeros(space.shape, dtype=space.dtype)
            return dummy_obs
        else:
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

    def _create_jit_feature_extractor(self):
        """创建 JIT 编译的特征提取函数"""

        # 为 Pi0 模型创建一个特征提取方法
        def extract_features(obs_dict):
            # 将观测转换为 Pi0 期望的格式
            observation = _model.Observation.from_dict(obs_dict)

            # 调用 Pi0 的特征提取
            # 这里假设我们修改了 Pi0 添加了 extract_features 方法
            # 或者我们直接调用其内部方法
            prefix_tokens, prefix_mask, prefix_ar_mask = self.pi0_model.embed_prefix(observation)

            # 返回聚合后的特征（例如平均池化）
            # 这里简化处理，实际可能需要更复杂的聚合
            features = jnp.mean(prefix_tokens, axis=1)  # [batch, feature_dim]
            return features

        # JIT 编译
        return nnx_utils.module_jit(extract_features)

    def _extract_pi0_features(self, obs: Union[np.ndarray, Dict]) -> np.ndarray:
        """从 Pi0 模型提取特征"""
        # 应用数据转换
        if isinstance(obs, dict):
            obs = self.transforms(obs)

        # 转换为 JAX 数组
        obs_jax = jax.tree_map(lambda x: jnp.asarray(x), obs)

        if self.freeze_pi0:
            # 使用冻结的 JIT 版本
            features_jax = self._pi0_extract_features(obs_jax)
        else:
            # 动态调用（允许梯度传播）
            observation = _model.Observation.from_dict(obs_jax)
            prefix_tokens, _, _ = self.pi0_model.embed_prefix(observation)
            features_jax = jnp.mean(prefix_tokens, axis=1)

        # 转换回 NumPy
        features_np = np.asarray(features_jax)
        return features_np

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            observations: 输入观测 (PyTorch tensor)

        Returns:
            features: 提取的特征 (PyTorch tensor)
        """
        # 如果输入是字典格式的观测
        if isinstance(observations, dict):
            # 批量处理
            batch_size = len(next(iter(observations.values())))
            features_list = []

            for i in range(batch_size):
                # 提取单个样本
                single_obs = {k: v[i].cpu().numpy() for k, v in observations.items()}
                # 提取 Pi0 特征
                pi0_features = self._extract_pi0_features(single_obs)
                features_list.append(pi0_features)

            # 堆叠并转换为 PyTorch tensor
            features_np = np.stack(features_list)
            features = torch.from_numpy(features_np).to(observations[list(observations.keys())[0]].device)
        else:
            # 处理普通 tensor 输入
            obs_np = observations.cpu().numpy()
            features_np = np.stack([self._extract_pi0_features(o) for o in obs_np])
            features = torch.from_numpy(features_np).to(observations.device)

        # 通过投影层
        features = self.feature_proj(features.float())
        return features


class Pi0ActorCriticPolicy(ActorCriticPolicy):
    """
    自定义 Actor-Critic 策略，使用 Pi0 作为 Actor 的初始化
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            pi0_model: Pi0 = None,
            pi0_checkpoint_path: str = None,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            freeze_pi0_features: bool = False,
            use_pi0_action_head: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        """
        Args:
            observation_space: 观测空间
            action_space: 动作空间
            lr_schedule: 学习率调度
            pi0_model: Pi0 模型实例
            pi0_checkpoint_path: Pi0 检查点路径（如果没有提供模型实例）
            features_extractor_kwargs: 特征提取器的额外参数
            freeze_pi0_features: 是否冻结 Pi0 特征提取部分
            use_pi0_action_head: 是否使用 Pi0 的动作头初始化
            normalize_images: 是否归一化图像
        """
        self.pi0_model = pi0_model
        self.pi0_checkpoint_path = pi0_checkpoint_path
        self.freeze_pi0_features = freeze_pi0_features
        self.use_pi0_action_head = use_pi0_action_head

        # 加载 Pi0 模型（如果需要）
        if self.pi0_model is None and self.pi0_checkpoint_path:
            self.pi0_model = self._load_pi0_model(pi0_checkpoint_path)

        # 设置特征提取器参数
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        features_extractor_kwargs.update({
            "pi0_model": self.pi0_model,
            "freeze_pi0": freeze_pi0_features,
        })

        # 调用父类构造函数
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=Pi0FeatureExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs
        )

        # 禁用正交初始化，保留预训练权重
        self.ortho_init = False

        # 使用 Pi0 的动作头初始化 Actor 网络
        if use_pi0_action_head and hasattr(self.pi0_model, 'action_out_proj'):
            self._init_action_net_from_pi0()

        # 初始化 log_std 为较小值，减少初始探索
        if hasattr(self, "log_std"):
            nn.init.constant_(self.log_std, -2.0)  # std ≈ 0.14

    def _load_pi0_model(self, checkpoint_path: str) -> Pi0:
        """从检查点加载 Pi0 模型"""
        # 这里需要实现实际的加载逻辑
        # 参考 openpi 的 checkpoint 加载方式
        from openpi.training import checkpoints
        from openpi.training import config as training_config

        # 加载配置和模型
        # 这是简化版本，实际需要根据项目结构调整
        raise NotImplementedError("需要实现 Pi0 模型加载逻辑")

    def _init_action_net_from_pi0(self):
        """使用 Pi0 的动作头权重初始化 Actor 网络"""
        # 获取 Pi0 的动作输出层参数
        if hasattr(self.pi0_model, 'action_out_proj'):
            # 获取 JAX/Flax 参数
            pi0_action_params = self.pi0_model.action_out_proj

            # 转换为 PyTorch 格式
            # 注意：需要处理 JAX 到 PyTorch 的转换
            # 这里是简化示例
            with torch.no_grad():
                # 假设可以访问权重和偏置
                if hasattr(pi0_action_params, 'kernel'):
                    weight_np = np.asarray(pi0_action_params.kernel.value)
                    # JAX 使用 (in, out)，PyTorch 使用 (out, in)
                    weight_np = weight_np.T
                    self.action_net.weight.data = torch.from_numpy(weight_np)

                if hasattr(pi0_action_params, 'bias'):
                    bias_np = np.asarray(pi0_action_params.bias.value)
                    self.action_net.bias.data = torch.from_numpy(bias_np)

    def _build_mlp_extractor(self) -> None:
        """构建 MLP 提取器"""
        # 父类会调用这个方法
        # 由于我们在 __init__ 中已经设置了 features_extractor_class
        # 这里主要是为了兼容性
        super()._build_mlp_extractor()

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            actions, values, log_probs
        """
        # 提取特征
        features = self.extract_features(obs)

        # 如果特征提取器输出是元组（actor特征，critic特征）
        if isinstance(features, tuple):
            latent_pi, latent_vf = features
        else:
            # 共享特征
            latent_pi = features
            latent_vf = features.detach()  # Critic 使用 detached 特征

        # 通过策略网络
        if self.mlp_extractor is not None:
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # 获取动作分布
        distribution = self._get_action_dist_from_latent(latent_pi)

        # 采样或选择确定性动作
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        # 计算价值
        values = self.value_net(latent_vf)

        # 计算对数概率
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作

        Returns:
            values, log_prob, entropy
        """
        # 提取特征
        features = self.extract_features(obs)

        if isinstance(features, tuple):
            latent_pi, latent_vf = features
        else:
            latent_pi = features
            latent_vf = features

        # 通过网络
        if self.mlp_extractor is not None:
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # 获取分布
        distribution = self._get_action_dist_from_latent(latent_pi)

        # 计算对数概率
        log_prob = distribution.log_prob(actions)

        # 计算价值
        values = self.value_net(latent_vf)

        # 计算熵
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor) -> DiagGaussianDistribution:
        """获取动作分布"""
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            latent_pi, _ = features
        else:
            latent_pi = features

        if self.mlp_extractor is not None:
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)

        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """预测状态价值"""
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            _, latent_vf = features
        else:
            latent_vf = features

        if self.mlp_extractor is not None:
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        return self.value_net(latent_vf)


class HybridPi0Policy(Pi0ActorCriticPolicy):
    """
    混合策略：结合 Pi0 的确定性策略和 PPO 的随机策略
    用于更稳定的微调
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            pi0_blend_ratio: float = 0.5,
            pi0_blend_decay: float = 0.99,
            **kwargs
    ):
        """
        Args:
            pi0_blend_ratio: Pi0 策略的初始混合比例 (0-1)
            pi0_blend_decay: 混合比例的衰减率
        """
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.pi0_blend_ratio = pi0_blend_ratio
        self.pi0_blend_decay = pi0_blend_decay
        self.current_blend_ratio = pi0_blend_ratio

        # 创建 Pi0 动作推理函数
        self._pi0_predict = self._create_pi0_predictor()

    def _create_pi0_predictor(self):
        """创建 Pi0 的确定性动作预测器"""

        def predict_pi0_action(obs_dict):
            # 使用 Pi0 模型预测动作
            observation = _model.Observation.from_dict(obs_dict)
            # 这里需要 Pi0 提供确定性预测接口
            actions = self.pi0_model.sample_actions(
                jax.random.PRNGKey(0),
                observation,
                num_steps=1  # 单步预测
            )
            return actions

        return nnx_utils.module_jit(predict_pi0_action)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        混合前向传播：结合 Pi0 和 PPO 策略
        """
        # 获取 PPO 策略输出
        ppo_actions, values, log_probs = super().forward(obs, deterministic)

        # 如果在训练模式且混合比例 > 0
        if self.training and self.current_blend_ratio > 0:
            # 获取 Pi0 确定性动作
            with torch.no_grad():
                if isinstance(obs, dict):
                    obs_dict = {k: v.cpu().numpy() for k, v in obs.items()}
                else:
                    obs_dict = {"state": obs.cpu().numpy()}

                pi0_actions_np = np.asarray(self._pi0_predict(obs_dict))
                pi0_actions = torch.from_numpy(pi0_actions_np).to(ppo_actions.device)

            # 混合动作
            blended_actions = (
                    self.current_blend_ratio * pi0_actions +
                    (1 - self.current_blend_ratio) * ppo_actions
            )

            # 重新计算混合动作的对数概率
            distribution = self.get_distribution(obs)
            log_probs = distribution.log_prob(blended_actions)

            return blended_actions, values, log_probs

        return ppo_actions, values, log_probs

    def update_blend_ratio(self):
        """更新混合比例（在每个 epoch 后调用）"""
        self.current_blend_ratio *= self.pi0_blend_decay


def create_pi0_policy_kwargs(*,
                             pi0_model,
                             transforms,
                             features_dim=256,
                             freeze_pi0=False,
                             use_hybrid=False):
    """
    返回 SB3 Policy 的类和 kwargs
    """
    from stable_baselines3.common.torch_layers import ActorCriticPolicy

    # ❶ 先定义 feature extractor
    def feature_extractor_factory(obs_space):
        return Pi0FeatureExtractor(
            observation_space=obs_space,
            pi0_model=pi0_model,
            transforms=transforms,
            features_dim=features_dim,
        )

    # ❷ 再组装 policy kwargs
    policy_kwargs = dict(
        features_extractor_class=feature_extractor_factory,
        features_extractor_kwargs={},     # 工厂不用额外 kwargs
    )
    # 其余 frozen 参数、混合策略逻辑保持原状
    return ActorCriticPolicy, policy_kwargs