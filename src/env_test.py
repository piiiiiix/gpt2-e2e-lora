import torch, transformers, datasets, peft

print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('datasets:', datasets.__version__)
print('peft:', peft.__version__)
print('cuda:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0))