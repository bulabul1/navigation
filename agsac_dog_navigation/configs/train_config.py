"""
训练配置文件
使用dataclass定义所有训练参数
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import json


@dataclass
class EnvironmentConfig:
    """环境配置"""
    max_pedestrians: int = 10
    max_corridors: int = 10
    max_vertices: int = 20
    obs_horizon: int = 8
    pred_horizon: int = 12
    max_episode_steps: int = 100
    use_geometric_reward: bool = True
    
    # Corridor生成器配置
    use_corridor_generator: bool = False
    curriculum_learning: bool = False
    scenario_seed: Optional[int] = None
    
    # Corridor约束配置
    corridor_constraint_mode: str = 'soft'  # 'soft', 'medium', 'hard'
    
    # 奖励函数微调参数
    corridor_penalty_weight: float = 10.0
    corridor_penalty_cap: float = 30.0
    progress_reward_weight: float = 20.0
    step_penalty_weight: float = 0.01
    enable_step_limit: bool = True
    
    # 设备
    device: str = 'cpu'


@dataclass
class ModelConfig:
    """模型配置"""
    action_dim: int = 22  # 11个路径点
    hidden_dim: int = 128
    num_modes: int = 3  # 多模态轨迹预测数量
    # SAC 熵策略
    auto_entropy: bool = True
    
    # 与环境一致的配置（会自动同步）
    max_pedestrians: int = 10
    max_corridors: int = 10
    max_vertices: int = 20
    obs_horizon: int = 8
    pred_horizon: int = 12
    
    # 预训练权重
    use_pretrained_predictor: bool = True
    pretrained_weights_path: Optional[str] = "external/SocialCircle_original/weights/SocialCircle/evsczara1"
    
    # 设备
    device: str = 'cpu'


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    episodes: int = 100
    warmup_episodes: int = 10
    updates_per_episode: int = 10
    
    # Buffer和批量
    buffer_capacity: int = 5000
    seq_len: int = 16
    batch_size: int = 16
    
    # 学习率
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    encoder_lr: Optional[float] = None  # 编码器学习率，None则使用critic_lr
    
    # SAC参数
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    
    # 日志和保存
    log_interval: int = 10
    save_interval: int = 50
    eval_interval: int = 50
    use_tensorboard: bool = True
    
    # 实验名称
    experiment_name: str = "agsac_training"


@dataclass
class AGSACConfig:
    """完整的AGSAC训练配置"""
    # 训练模式
    mode: str = 'fixed'  # fixed, dynamic, curriculum
    
    # 子配置
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 全局设置
    seed: Optional[int] = None
    device: str = 'cpu'
    
    def __post_init__(self):
        """初始化后处理：确保配置一致性"""
        # 根据模式自动配置环境
        if self.mode == 'fixed':
            self.env.use_corridor_generator = False
            self.env.curriculum_learning = False
        elif self.mode == 'dynamic':
            self.env.use_corridor_generator = True
            self.env.curriculum_learning = False
            self.env.scenario_seed = self.seed
        elif self.mode == 'curriculum':
            self.env.use_corridor_generator = True
            self.env.curriculum_learning = True
            self.env.scenario_seed = self.seed
        
        # 同步设备配置
        self.env.device = self.device
        self.model.device = self.device
        
        # 同步场景参数（确保环境和模型一致）
        self.model.max_pedestrians = self.env.max_pedestrians
        self.model.max_corridors = self.env.max_corridors
        self.model.max_vertices = self.env.max_vertices
        self.model.obs_horizon = self.env.obs_horizon
        self.model.pred_horizon = self.env.pred_horizon
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_yaml(self, path: Path):
        """保存为YAML文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, path: Path):
        """保存为JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AGSACConfig':
        """从字典创建"""
        env_config = EnvironmentConfig(**config_dict.get('env', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            mode=config_dict.get('mode', 'fixed'),
            env=env_config,
            model=model_config,
            training=training_config,
            seed=config_dict.get('seed'),
            device=config_dict.get('device', 'cpu')
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'AGSACConfig':
        """从YAML文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: Path) -> 'AGSACConfig':
        """从JSON文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

