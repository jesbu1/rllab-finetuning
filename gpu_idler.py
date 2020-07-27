import torch
import time
large_matrix = torch.rand((500, 500, 500)).cuda()
other_large_matrix = torch.rand((500, 500, 500)).cuda()
while True:
    time.sleep(1)
    result = other_large_matrix.clone()
    for i in range(5):
        result = torch.matmul(large_matrix, result)
    print(result.mean())
