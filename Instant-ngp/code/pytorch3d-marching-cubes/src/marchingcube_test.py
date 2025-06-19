import torch
import numpy as np
from marching_cubes import create_scalar_field  # 假设该函数在marching_cubes.py中定义

def test_create_scalar_field():
    dim_x, dim_y, dim_z = 32, 32, 32
    model = lambda x: torch.sin(x.sum(dim=1))  # 示例模型，实际模型应替换
    scalar_field = create_scalar_field(dim_x, dim_y, dim_z, model)

    assert scalar_field.shape == (dim_x, dim_y, dim_z), "Scalar field shape mismatch"
    assert isinstance(scalar_field, np.ndarray), "Scalar field should be a numpy array"

def run_tests():
    test_create_scalar_field()
    print("所有测试通过！")

if __name__ == "__main__":
    run_tests()