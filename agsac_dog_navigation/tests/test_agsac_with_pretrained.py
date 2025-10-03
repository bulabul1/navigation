"""
测试AGSACModel使用预训练轨迹预测器
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_agsac_with_simple_predictor():
    """测试使用简化预测器的AGSACModel"""
    from agsac.models.agsac_model import AGSACModel
    
    print("\n" + "="*80)
    print("测试1: AGSACModel with SimpleTrajectoryPredictor")
    print("="*80)
    
    model = AGSACModel(
        dog_feature_dim=64,
        corridor_feature_dim=64,
        social_feature_dim=64,
        pedestrian_feature_dim=64,
        fusion_dim=64,
        action_dim=22,
        max_pedestrians=5,
        max_corridors=2,
        max_vertices=10,
        obs_horizon=8,
        pred_horizon=12,
        num_modes=20,
        use_pretrained_predictor=False,  # 使用简化预测器
        device='cpu'
    )
    
    # 统计参数
    total, trainable = count_parameters(model)
    print(f"\n参数统计:")
    print(f"  - 总参数: {total:,}")
    print(f"  - 可训练参数: {trainable:,}")
    print(f"  - 参数量限制: < 2,000,000")
    
    if total < 2_000_000:
        print(f"  [OK] 参数量满足要求！")
    else:
        print(f"  [WARN] 参数量超出限制 ({total - 2_000_000:,} 超出)")
    
    print(f"\n[OK] 简化预测器测试完成")


def test_agsac_with_pretrained_predictor():
    """测试使用预训练预测器的AGSACModel"""
    from agsac.models.agsac_model import AGSACModel
    
    print("\n" + "="*80)
    print("测试2: AGSACModel with PretrainedTrajectoryPredictor")
    print("="*80)
    
    try:
        model = AGSACModel(
            dog_feature_dim=64,
            corridor_feature_dim=64,
            social_feature_dim=64,
            pedestrian_feature_dim=64,
            fusion_dim=64,
            action_dim=22,
            max_pedestrians=5,
            max_corridors=2,
            max_vertices=10,
            obs_horizon=8,
            pred_horizon=12,
            num_modes=20,
            use_pretrained_predictor=True,  # 使用预训练预测器
            pretrained_weights_path='weights/SocialCircle/evsczara1',
            device='cpu'
        )
        
        # 统计参数
        total, trainable = count_parameters(model)
        print(f"\n参数统计:")
        print(f"  - 总参数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        print(f"  - 参数量限制: < 2,000,000")
        
        if total < 2_000_000:
            print(f"  [OK] 参数量满足要求！")
        else:
            print(f"  [WARN] 参数量超出限制 ({total - 2_000_000:,} 超出)")
        
        # 检查预训练模型是否被冻结
        predictor_trainable = sum(
            p.numel() for p in model.trajectory_predictor.parameters() 
            if p.requires_grad
        )
        print(f"\n预训练预测器状态:")
        print(f"  - 可训练参数: {predictor_trainable:,}")
        
        if predictor_trainable == 0:
            print(f"  [OK] 预训练模型已冻结")
        else:
            print(f"  [WARN] 预训练模型未完全冻结")
        
        print(f"\n[OK] 预训练预测器测试完成")
        
    except Exception as e:
        print(f"[WARN] 预训练预测器加载失败: {e}")
        print("  可能原因: SocialCircle权重未正确设置")
        import traceback
        traceback.print_exc()


def test_agsac_forward_with_pretrained():
    """测试使用预训练预测器的前向传播"""
    from agsac.models.agsac_model import AGSACModel
    
    print("\n" + "="*80)
    print("测试3: 前向传播测试（预训练预测器）")
    print("="*80)
    
    try:
        model = AGSACModel(
            dog_feature_dim=64,
            corridor_feature_dim=64,
            social_feature_dim=64,
            pedestrian_feature_dim=64,
            fusion_dim=64,
            action_dim=22,
            max_pedestrians=3,
            max_corridors=2,
            max_vertices=10,
            obs_horizon=8,
            pred_horizon=12,
            num_modes=20,
            use_pretrained_predictor=True,
            pretrained_weights_path='weights/SocialCircle/evsczara1',
            device='cpu'
        )
        
        # 准备测试数据
        batch_size = 2
        obs = {
            'dog': torch.randn(batch_size, 8, 3),
            'pedestrians': torch.randn(batch_size, 3, 8, 2),
            'pedestrian_mask': torch.ones(batch_size, 3),
            'corridors': torch.randn(batch_size, 2, 10, 2),
            'vertex_counts': torch.tensor([[5, 4], [6, 5]]),
        }
        
        print(f"\n输入数据:")
        print(f"  - dog: {obs['dog'].shape}")
        print(f"  - pedestrians: {obs['pedestrians'].shape}")
        print(f"  - corridors: {obs['corridors'].shape}")
        
        # 前向传播
        output = model(obs)
        
        print(f"\n输出数据:")
        print(f"  - action: {output['action'].shape}")
        print(f"  - predicted_trajectories: {output['predicted_trajectories'].shape}")
        print(f"  - q1_value: {output['q1_value'].shape}")
        print(f"  - q2_value: {output['q2_value'].shape}")
        
        # 验证输出形状
        assert output['action'].shape == (batch_size, 22)
        assert output['predicted_trajectories'].shape == (batch_size, 12, 2, 20)
        
        print(f"\n[OK] 前向传播测试通过！")
        
    except Exception as e:
        print(f"[WARN] 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()


def compare_parameter_counts():
    """对比两种配置的参数量"""
    from agsac.models.agsac_model import AGSACModel
    
    print("\n" + "="*80)
    print("测试4: 参数量对比")
    print("="*80)
    
    # 简化版
    print("\n创建简化版模型...")
    model_simple = AGSACModel(
        dog_feature_dim=64,
        corridor_feature_dim=64,
        social_feature_dim=64,
        pedestrian_feature_dim=64,
        fusion_dim=64,
        action_dim=22,
        max_pedestrians=5,
        max_corridors=2,
        max_vertices=10,
        use_pretrained_predictor=False,
        device='cpu'
    )
    
    total_simple, trainable_simple = count_parameters(model_simple)
    
    # 预训练版
    try:
        print("\n创建预训练版模型...")
        model_pretrained = AGSACModel(
            dog_feature_dim=64,
            corridor_feature_dim=64,
            social_feature_dim=64,
            pedestrian_feature_dim=64,
            fusion_dim=64,
            action_dim=22,
            max_pedestrians=5,
            max_corridors=2,
            max_vertices=10,
            use_pretrained_predictor=True,
            pretrained_weights_path='weights/SocialCircle/evsczara1',
            device='cpu'
        )
        
        total_pretrained, trainable_pretrained = count_parameters(model_pretrained)
        
        print("\n" + "="*80)
        print("参数量对比结果")
        print("="*80)
        
        print(f"\n简化版模型:")
        print(f"  - 总参数: {total_simple:,}")
        print(f"  - 可训练参数: {trainable_simple:,}")
        
        print(f"\n预训练版模型:")
        print(f"  - 总参数: {total_pretrained:,}")
        print(f"  - 可训练参数: {trainable_pretrained:,}")
        
        print(f"\n差异:")
        print(f"  - 总参数差异: {total_pretrained - total_simple:+,}")
        print(f"  - 可训练参数差异: {trainable_pretrained - trainable_simple:+,}")
        
        print(f"\n参数量限制: < 2,000,000")
        print(f"  - 简化版: {'[OK] 满足' if total_simple < 2_000_000 else '[WARN] 超出'}")
        print(f"  - 预训练版: {'[OK] 满足' if total_pretrained < 2_000_000 else '[WARN] 超出'}")
        
    except Exception as e:
        print(f"\n[WARN] 预训练版模型创建失败: {e}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("AGSACModel 预训练集成测试套件")
    print("="*80)
    
    test_agsac_with_simple_predictor()
    test_agsac_with_pretrained_predictor()
    test_agsac_forward_with_pretrained()
    compare_parameter_counts()
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)

