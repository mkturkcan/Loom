"""
Extended ISA ONNX Export
========================

Exports the Extended ISA Neural Computer to ONNX for browser execution.
Uses a module wrapper to register transformer weights as buffers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extended_isa_v4 import (
    ExtendedNeuralComputerV4, ExtendedConfigV4,
    init_state_v4, read_memory_v4, get_pc_v4,
    OP_HALT, OP_INC, OP_DEC, OP_MOV, OP_ADD, OP_SUB
)
from subleq import to_bipolar, from_bipolar, signed_to_bipolar, signed_from_bipolar


class ExtendedISAModule(nn.Module):
    """PyTorch Module wrapper for Extended ISA with fixed weights."""
    
    def __init__(self, cfg: ExtendedConfigV4):
        super().__init__()
        self.cfg = cfg
        
        # Build the computer with fixed weights
        self.computer = ExtendedNeuralComputerV4(cfg)
        
        # Convert all layer weights to registered buffers (frozen)
        self.layer_data = nn.ModuleList()
        
        for i, layer in enumerate(self.computer.layers):
            layer_module = nn.Module()
            
            # Register Q, K, V matrices based on number of heads
            if hasattr(layer, 'Q') and layer.Q is not None:
                layer_module.register_buffer('Q', layer.Q)
                layer_module.register_buffer('K', layer.K)
                layer_module.register_buffer('V', layer.V)
            
            if hasattr(layer, 'Q1') and layer.Q1 is not None:
                layer_module.register_buffer('Q1', layer.Q1)
                layer_module.register_buffer('K1', layer.K1)
                layer_module.register_buffer('V1', layer.V1)
            
            if hasattr(layer, 'Q2') and layer.Q2 is not None:
                layer_module.register_buffer('Q2', layer.Q2)
                layer_module.register_buffer('K2', layer.K2)
                layer_module.register_buffer('V2', layer.V2)
            
            if hasattr(layer, 'Q3') and layer.Q3 is not None:
                layer_module.register_buffer('Q3', layer.Q3)
                layer_module.register_buffer('K3', layer.K3)
                layer_module.register_buffer('V3', layer.V3)
            
            # Register FFN weights
            layer_module.register_buffer('W1', layer.W1)
            layer_module.register_buffer('b1', layer.b1)
            layer_module.register_buffer('W2', layer.W2)
            layer_module.register_buffer('b2', layer.b2)
            layer_module.num_heads = layer.num_heads
            layer_module.lam = layer.lam
            
            self.layer_data.append(layer_module)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Execute one instruction through 10 transformer layers."""
        for layer in self.layer_data:
            X = self._layer_forward(X, layer)
        return X
    
    def _layer_forward(self, X: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """Forward pass for one transformer layer."""
        lam = layer.lam
        
        if layer.num_heads == 1:
            Q, K, V = layer.Q, layer.K, layer.V
            scores = torch.mm(torch.mm(X.t(), K.t()), torch.mm(Q, X))
            attn_weights = F.softmax(lam * scores, dim=0)
            attn = X + torch.mm(V, torch.mm(X, attn_weights))
        elif layer.num_heads == 2:
            Q1, K1, V1 = layer.Q1, layer.K1, layer.V1
            Q2, K2, V2 = layer.Q2, layer.K2, layer.V2
            scores1 = torch.mm(torch.mm(X.t(), K1.t()), torch.mm(Q1, X))
            scores2 = torch.mm(torch.mm(X.t(), K2.t()), torch.mm(Q2, X))
            attn_weights1 = F.softmax(lam * scores1, dim=0)
            attn_weights2 = F.softmax(lam * scores2, dim=0)
            attn = X + torch.mm(V1, torch.mm(X, attn_weights1)) + \
                       torch.mm(V2, torch.mm(X, attn_weights2))
        else:  # 3 heads
            Q1, K1, V1 = layer.Q1, layer.K1, layer.V1
            Q2, K2, V2 = layer.Q2, layer.K2, layer.V2
            Q3, K3, V3 = layer.Q3, layer.K3, layer.V3
            scores1 = torch.mm(torch.mm(X.t(), K1.t()), torch.mm(Q1, X))
            scores2 = torch.mm(torch.mm(X.t(), K2.t()), torch.mm(Q2, X))
            scores3 = torch.mm(torch.mm(X.t(), K3.t()), torch.mm(Q3, X))
            attn_weights1 = F.softmax(lam * scores1, dim=0)
            attn_weights2 = F.softmax(lam * scores2, dim=0)
            attn_weights3 = F.softmax(lam * scores3, dim=0)
            attn = X + torch.mm(V1, torch.mm(X, attn_weights1)) + \
                       torch.mm(V2, torch.mm(X, attn_weights2)) + \
                       torch.mm(V3, torch.mm(X, attn_weights3))
        
        # FFN
        ff1 = F.relu(torch.mm(layer.W1, attn) + layer.b1)
        output = attn + torch.mm(layer.W2, ff1) + layer.b2
        
        return output


def test_operations():
    """Test that basic operations work."""
    print("Testing Extended ISA operations...")
    
    cfg = ExtendedConfigV4(s=32, m=8, n=64, N=8)
    
    # Test INC
    memory = [5, 0, 0, 0, 0, 0, 0, 0]
    commands = [(OP_INC, cfg.s + 0, 0), (OP_HALT, 0, 0)]
    
    X = init_state_v4(cfg, memory, commands)
    computer = ExtendedNeuralComputerV4(cfg)
    
    with torch.no_grad():
        X = computer.step(X)
    
    result = read_memory_v4(X, cfg)
    success = (result[0] == 6)
    print(f"  INC 5 -> {result[0]} (expected 6): {'PASS' if success else 'FAIL'}")
    
    return success


def export_onnx():
    """Export Extended ISA to ONNX."""
    print("=" * 60)
    print("Extended ISA - ONNX Export")
    print("=" * 60)
    
    # Test first
    if not test_operations():
        print("ERROR: Basic operation test failed!")
        return False
    
    print("\nExporting to ONNX...")
    
    # Create configuration
    cfg = ExtendedConfigV4(s=32, m=64, n=1024, N=8)
    
    print(f"\nConfiguration:")
    print(f"  d_model: {cfg.d_model}")
    print(f"  n_cols: {cfg.n}")
    print(f"  Memory slots: {cfg.m}")
    print(f"  State size: {cfg.d_model * cfg.n * 4 / 1024:.1f} KB")
    
    # Create module
    model = ExtendedISAModule(cfg)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Simple demo program
    memory = [0] * cfg.m
    memory[0] = 0  # x position
    memory[1] = 0  # y position
    memory[2] = 0  # direction
    memory[3] = 5  # food x
    memory[4] = 3  # food y
    memory[5] = 0  # score
    memory[6] = 0  # frame
    
    # Simple loop: INC x, INC frame, loop
    cmd_start = cfg.s + cfg.m
    commands = [
        (OP_INC, cfg.s + 0, 0),  # x++
        (OP_INC, cfg.s + 6, 0),  # frame++
        (OP_HALT, 0, 0)  # halt for demo
    ]
    
    X = init_state_v4(cfg, memory, commands)
    
    # Export
    output_dir = os.path.dirname(__file__)
    onnx_path = os.path.join(output_dir, 'snake.onnx')
    
    print(f"\nExporting to {onnx_path}...")
    
    # Use legacy export if available
    os.environ['PYTORCH_ONNX_USE_LEGACY_EXPORT'] = '1'
    
    torch.onnx.export(
        model,
        X,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['new_state'],
        verbose=False
    )
    
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Saved: {size_mb:.1f} MB")
    
    # Save config
    config = {
        's': cfg.s,
        'm': cfg.m,
        'n': cfg.n,
        'N': cfg.N,
        'logn': cfg.logn,
        'd_model': cfg.d_model,
        'idx_memory': cfg.idx_memory,
        'idx_pc': cfg.idx_pc,
        'head_x_addr': 0,
        'head_y_addr': 1,
        'direction_addr': 2,
        'food_x_addr': 3,
        'food_y_addr': 4,
        'score_addr': 5,
        'frame_addr': 6,
        'game_over_addr': 7,
        'input_addr': 8,
        'length_addr': 9,
        'grid_size': 7,
        'isa_type': 'extended'
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")
    
    # Save initial state
    with open(os.path.join(output_dir, 'initial_state.json'), 'w') as f:
        json.dump({'state': X.tolist(), 'memory': memory}, f)
    print("Saved initial_state.json")
    
    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        
        result = session.run(None, {'state': X.numpy()})
        X_onnx = torch.from_numpy(result[0])
        
        X_torch = model(X)
        diff = torch.abs(X_onnx - X_torch).max().item()
        print(f"ONNX verification: OK (max diff = {diff:.6f})")
        
    except Exception as e:
        print(f"ONNX verification failed: {e}")
    
    print("\nExport complete!")
    return True


if __name__ == "__main__":
    export_onnx()
