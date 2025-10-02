"""测试MultiModalFusion模块"""
import torch
import pytest
from agsac.models.fusion.multi_modal_fusion import (
    MultiModalFusion, 
    SimplifiedFusion,
    create_fusion_module
)


def test_fusion_forward_single():
    """测试单个样本的前向传播"""
    fusion = MultiModalFusion()
    
    dog = torch.randn(64)
    ped = torch.randn(64)
    corridor = torch.randn(128)
    
    fused, attn = fusion(dog, ped, corridor, return_attention_weights=True)
    
    assert fused.shape == (64,), f"输出维度错误，期望(64,)，实际{fused.shape}"
    assert attn.shape == (1, 2), f"注意力权重维度错误，期望(1, 2)，实际{attn.shape}"
    # 注意：训练模式下，由于dropout，注意力权重可能不完全归一化
    # 这是正常现象，不影响功能


def test_fusion_forward_batch():
    """测试batch输入"""
    fusion = MultiModalFusion()
    batch_size = 16
    
    dog = torch.randn(batch_size, 64)
    ped = torch.randn(batch_size, 64)
    corridor = torch.randn(batch_size, 128)
    
    fused, attn = fusion(dog, ped, corridor, return_attention_weights=True)
    
    assert fused.shape == (batch_size, 64), "Batch输出维度错误"
    assert attn.shape == (batch_size, 1, 2), "Batch注意力权重维度错误"


def test_fusion_gradient_flow():
    """测试梯度传播"""
    fusion = MultiModalFusion()
    
    dog = torch.randn(64, requires_grad=True)
    ped = torch.randn(64, requires_grad=True)
    corridor = torch.randn(128, requires_grad=True)
    
    fused, _ = fusion(dog, ped, corridor)
    loss = fused.sum()
    loss.backward()
    
    assert dog.grad is not None, "dog特征梯度为空"
    assert ped.grad is not None, "ped特征梯度为空"
    assert corridor.grad is not None, "corridor特征梯度为空"
    assert torch.isfinite(dog.grad).all(), "dog梯度包含nan或inf"
    assert torch.isfinite(ped.grad).all(), "ped梯度包含nan或inf"
    assert torch.isfinite(corridor.grad).all(), "corridor梯度包含nan或inf"


def test_fusion_parameter_count():
    """测试参数量"""
    fusion = MultiModalFusion()
    total_params = sum(p.numel() for p in fusion.parameters())
    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    
    print(f"\nFusion模块参数量: {total_params:,} (可训练: {trainable_params:,})")
    assert total_params < 60000, f"参数量{total_params}超过预期(60K)"
    assert trainable_params == total_params, "所有参数应该可训练"


def test_attention_weights_sum_to_one():
    """测试注意力权重归一化"""
    fusion = MultiModalFusion()
    fusion.eval()  # 关闭dropout，确保一致性
    
    with torch.no_grad():
        for _ in range(10):
            dog = torch.randn(64)
            ped = torch.randn(64)
            corridor = torch.randn(128)
            
            _, attn = fusion(dog, ped, corridor, return_attention_weights=True)
            attn_sum = attn.sum(dim=-1)
            assert torch.allclose(attn_sum, torch.tensor([[1.0]]), atol=1e-5), \
                f"注意力权重和应为1，实际为{attn_sum}"


def test_simplified_fusion():
    """测试简化版融合"""
    simple = SimplifiedFusion()
    
    dog = torch.randn(64)
    ped = torch.randn(64)
    corridor = torch.randn(128)
    
    fused, attn = simple(dog, ped, corridor, return_attention_weights=True)
    
    assert fused.shape == (64,), "简化版输出维度错误"
    assert attn is None, "简化版不应返回注意力权重"
    
    # 测试参数量
    params = sum(p.numel() for p in simple.parameters())
    print(f"\n简化版Fusion参数量: {params:,}")
    assert params < 50000, f"简化版参数量{params}超过预期(50K)"


def test_factory_function():
    """测试工厂函数"""
    # 创建注意力版本
    attn_fusion = create_fusion_module('attention', num_heads=4)
    assert isinstance(attn_fusion, MultiModalFusion)
    
    # 创建简化版本
    simple_fusion = create_fusion_module('simple')
    assert isinstance(simple_fusion, SimplifiedFusion)
    
    # 测试两者都能正常工作
    dog = torch.randn(64)
    ped = torch.randn(64)
    corridor = torch.randn(128)
    
    out1, _ = attn_fusion(dog, ped, corridor)
    out2, _ = simple_fusion(dog, ped, corridor)
    
    assert out1.shape == out2.shape == (64,)


def test_eval_mode_deterministic():
    """测试eval模式下的确定性"""
    fusion = MultiModalFusion()
    fusion.eval()
    
    dog = torch.randn(64)
    ped = torch.randn(64)
    corridor = torch.randn(128)
    
    with torch.no_grad():
        out1, attn1 = fusion(dog, ped, corridor, return_attention_weights=True)
        out2, attn2 = fusion(dog, ped, corridor, return_attention_weights=True)
    
    assert torch.allclose(out1, out2), "eval模式下相同输入应产生相同输出"
    assert torch.allclose(attn1, attn2), "eval模式下注意力权重应一致"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])