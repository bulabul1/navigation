"""测试几何微分评估器 (GDE)"""
import numpy as np
import torch
import pytest
from agsac.models.evaluator import GeometricDifferentialEvaluator


def test_gde_basic():
    """测试基础前向传播"""
    gde = GeometricDifferentialEvaluator(eta=0.5, M=10)
    
    # 创建简单路径
    path = torch.zeros(11, 2)
    path[:, 0] = torch.linspace(0, 10, 11)
    reference_line = torch.tensor([1.0, 0.0])
    
    score = gde(path, reference_line)
    
    assert isinstance(score, torch.Tensor), "输出应该是tensor"
    assert score.shape == (), "输出应该是标量"
    assert 0.0 <= score.item() <= 1.0, f"评分应在[0, 1]范围，实际{score.item()}"


def test_gde_perfect_alignment():
    """测试完美对齐情况"""
    gde = GeometricDifferentialEvaluator()
    
    # 路径和参考线都沿x轴
    path = torch.zeros(11, 2)
    path[:, 0] = torch.linspace(0, 10, 11)
    reference_line = torch.tensor([1.0, 0.0])
    
    score = gde(path, reference_line)
    
    assert score.item() > 0.99, f"完美对齐应得接近1.0的评分，实际{score.item()}"


def test_gde_perpendicular():
    """测试垂直情况"""
    gde = GeometricDifferentialEvaluator()
    
    # 路径沿y轴，参考线沿x轴
    path = torch.zeros(11, 2)
    path[:, 1] = torch.linspace(0, 10, 11)
    reference_line = torch.tensor([1.0, 0.0])
    
    score = gde(path, reference_line)
    
    # 垂直应该得0.5分（90度 / 180度 = 0.5，然后1 - 0.5 = 0.5）
    assert 0.4 < score.item() < 0.6, f"垂直应得约0.5的评分，实际{score.item()}"


def test_gde_backward():
    """测试反向情况"""
    gde = GeometricDifferentialEvaluator()
    
    # 路径沿-x轴，参考线沿x轴
    path = torch.zeros(11, 2)
    path[:, 0] = torch.linspace(10, 0, 11)  # 反向
    reference_line = torch.tensor([1.0, 0.0])
    
    score = gde(path, reference_line)
    
    # 反向应该得接近0的评分（180度 / 180度 = 1，然后1 - 1 = 0）
    assert score.item() < 0.1, f"反向应得接近0的评分，实际{score.item()}"


def test_gde_batch():
    """测试批量处理"""
    gde = GeometricDifferentialEvaluator()
    
    # 创建3个不同的路径
    path1 = torch.zeros(11, 2)
    path1[:, 0] = torch.linspace(0, 10, 11)  # 对齐
    
    path2 = torch.zeros(11, 2)
    path2[:, 1] = torch.linspace(0, 10, 11)  # 垂直
    
    path3 = torch.zeros(11, 2)
    path3[:, 0] = torch.linspace(10, 0, 11)  # 反向
    
    batch_paths = torch.stack([path1, path2, path3])  # (3, 11, 2)
    batch_refs = torch.stack([torch.tensor([1.0, 0.0])] * 3)  # (3, 2)
    
    batch_scores = gde.batch_forward(batch_paths, batch_refs)
    
    assert batch_scores.shape == (3,), f"批量输出维度错误，期望(3,)，实际{batch_scores.shape}"
    assert batch_scores[0] > 0.99, "第1个应高分"
    assert 0.4 < batch_scores[1] < 0.6, "第2个应中等分"
    assert batch_scores[2] < 0.1, "第3个应低分"


def test_gde_gradient():
    """测试梯度流"""
    gde = GeometricDifferentialEvaluator()
    
    # 创建需要梯度的路径
    path = torch.zeros(11, 2, requires_grad=True)
    path.data[:, 0] = torch.linspace(0, 10, 11)
    path.data[:, 1] = torch.randn(11) * 0.1  # 添加扰动
    
    reference_line = torch.tensor([1.0, 0.0])
    
    score = gde(path, reference_line)
    score.backward()
    
    assert path.grad is not None, "路径应该有梯度"
    # 检查大部分梯度是有限的
    finite_ratio = torch.isfinite(path.grad).float().mean()
    assert finite_ratio > 0.8, f"大部分梯度应为有限值，实际比例{finite_ratio.item()}"


def test_gde_parameter_count():
    """测试参数量（应该为0）"""
    gde = GeometricDifferentialEvaluator()
    
    total_params = sum(p.numel() for p in gde.parameters())
    trainable_params = sum(p.numel() for p in gde.parameters() if p.requires_grad)
    
    print(f"\nGDE参数量: {total_params} (可训练: {trainable_params})")
    
    assert trainable_params == 0, "GDE不应有可训练参数（纯计算模块）"
    # 可能有buffer（weights），但不应有可训练参数


def test_gde_numerical_stability():
    """测试数值稳定性"""
    gde = GeometricDifferentialEvaluator()
    reference_line = torch.tensor([1.0, 0.0])
    
    # 测试极短路径
    path_short = torch.zeros(11, 2)
    path_short[:, 0] = torch.linspace(0, 0.001, 11)
    score_short = gde(path_short, reference_line)
    assert torch.isfinite(score_short), "极短路径应能正常处理"
    
    # 测试零长度路径（所有点重合）
    path_zero = torch.zeros(11, 2)
    score_zero = gde(path_zero, reference_line)
    assert torch.isfinite(score_zero), "零长度路径应能正常处理"
    
    # 测试很长的路径
    path_long = torch.zeros(11, 2)
    path_long[:, 0] = torch.linspace(0, 1000, 11)
    score_long = gde(path_long, reference_line)
    assert torch.isfinite(score_long), "很长路径应能正常处理"
    # 长度不应影响评分（方向一致）
    assert score_long.item() > 0.99, "长路径但方向一致应高分"


def test_gde_reward_bonus():
    """测试奖励bonus计算"""
    eta = 0.7
    gde = GeometricDifferentialEvaluator(eta=eta)
    
    path = torch.zeros(11, 2)
    path[:, 0] = torch.linspace(0, 10, 11)
    reference_line = torch.tensor([1.0, 0.0])
    
    score = gde(path, reference_line)
    reward_bonus = gde.compute_reward_bonus(path, reference_line)
    
    expected_bonus = eta * score
    assert torch.allclose(reward_bonus, expected_bonus), \
        f"Reward bonus应为eta*score，期望{expected_bonus.item():.4f}，实际{reward_bonus.item():.4f}"


def test_gde_evaluate_alignment_details():
    """测试详细评估功能"""
    gde = GeometricDifferentialEvaluator()
    
    path = torch.zeros(11, 2)
    path[:, 0] = torch.linspace(0, 10, 11)
    reference_line = torch.tensor([1.0, 0.0])
    
    details = gde.evaluate_alignment(path, reference_line, return_details=True)
    
    # 检查返回的详细信息
    assert 'geo_score' in details, "应包含geo_score"
    assert 'angles' in details, "应包含angles"
    assert 'weights' in details, "应包含weights"
    assert 'avg_angle' in details, "应包含avg_angle"
    assert 'angle_degrees' in details, "应包含angle_degrees"
    
    # 检查维度
    assert len(details['angles']) == 10, "应有10个角度（11个点-1）"
    assert len(details['weights']) == 10, "应有10个权重"
    assert len(details['angle_degrees']) == 10, "应有10个角度（度）"
    
    # 完美对齐的情况，所有角度应接近0
    assert details['avg_angle'] < 0.01, "完美对齐的平均角度应接近0"


def test_gde_weights():
    """测试权重属性"""
    gde = GeometricDifferentialEvaluator(M=10)
    
    weights = gde.weights
    
    # 检查权重维度
    assert weights.shape == (10,), f"权重维度错误，期望(10,)，实际{weights.shape}"
    
    # 检查权重是递减的
    for i in range(len(weights) - 1):
        assert weights[i] >= weights[i+1], f"权重应递减，但weights[{i}]={weights[i]} < weights[{i+1}]={weights[i+1]}"
    
    # 检查所有权重为正
    assert (weights > 0).all(), "所有权重应为正数"
    
    # 检查第一个权重接近1.0
    assert weights[0].item() > 0.99, f"第一个权重应接近1.0，实际{weights[0].item()}"


def test_gde_different_reference_directions():
    """测试不同的参考方向"""
    gde = GeometricDifferentialEvaluator()
    
    # 同一路径
    path = torch.zeros(11, 2)
    path[:, 0] = torch.linspace(0, 10, 11)
    path[:, 1] = torch.linspace(0, 10, 11)  # 沿45度方向
    
    # 不同参考方向
    ref_aligned = torch.tensor([1.0, 1.0])  # 对齐
    ref_perpendicular = torch.tensor([1.0, -1.0])  # 垂直
    ref_backward = torch.tensor([-1.0, -1.0])  # 反向
    
    score_aligned = gde(path, ref_aligned)
    score_perp = gde(path, ref_perpendicular)
    score_back = gde(path, ref_backward)
    
    print(f"\n对齐评分: {score_aligned.item():.4f}")
    print(f"垂直评分: {score_perp.item():.4f}")
    print(f"反向评分: {score_back.item():.4f}")
    
    # 对齐应该最高分
    assert score_aligned > score_perp, "对齐应比垂直得分高"
    assert score_aligned > score_back, "对齐应比反向得分高"


def test_gde_curved_path():
    """测试曲线路径"""
    gde = GeometricDifferentialEvaluator()
    
    # 创建弧形路径
    t = torch.linspace(0, np.pi/6, 11)
    path_curve = torch.zeros(11, 2)
    path_curve[:, 0] = torch.cos(t) * 10
    path_curve[:, 1] = torch.sin(t) * 10
    
    reference_line = torch.tensor([1.0, 0.0])
    
    score = gde(path_curve, reference_line)
    
    # 曲线路径应该得到中等偏上的评分（因为初始方向接近参考方向）
    assert 0.0 < score.item() < 1.0, f"曲线路径应得有效评分，实际{score.item()}"
    print(f"\n弧形路径评分: {score.item():.4f}")


def test_gde_consistency():
    """测试batch_forward和单独forward的一致性"""
    gde = GeometricDifferentialEvaluator()
    
    # 创建路径
    path1 = torch.zeros(11, 2)
    path1[:, 0] = torch.linspace(0, 10, 11)
    
    path2 = torch.zeros(11, 2)
    path2[:, 1] = torch.linspace(0, 10, 11)
    
    reference_line = torch.tensor([1.0, 0.0])
    
    # 单独计算
    score1 = gde(path1, reference_line)
    score2 = gde(path2, reference_line)
    
    # 批量计算
    batch_paths = torch.stack([path1, path2])
    batch_refs = torch.stack([reference_line, reference_line])
    batch_scores = gde.batch_forward(batch_paths, batch_refs)
    
    # 应该一致
    assert torch.allclose(batch_scores[0], score1, atol=1e-6), \
        "批量计算应与单独计算一致"
    assert torch.allclose(batch_scores[1], score2, atol=1e-6), \
        "批量计算应与单独计算一致"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

