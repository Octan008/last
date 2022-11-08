import glob
import torch
path = "./log/tensorf_anim_VM_traintest15/tensorf_anim_VM_traintest15_sh*.th"
paths = glob.glob(path)
print(paths)

for p in paths:
    t = torch.load(p)
    print(t)
    print(t["state_dict"].keys())
    print()

