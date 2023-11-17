import torch
import os
from einops import rearrange

# Create directory for data
# os.mkdir("rbc_data")

# Read data
file = "data21_101.pt"
data = torch.load(file) #(2000, 2, 256, 1792); guessing this is (time, velocity component, x, y)
if file != 'rbc_data':
    data = rearrange(data, 'N h w c-> N c h w')
print(f"Loaded Data shape:{data.shape}")

# Standardization
velocity_norm = torch.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
std = torch.std(velocity_norm)  # Standard deviation of velocity norm
avg = torch.mean(data, dim=(0, 2, 3), keepdim=True)  # Mean velocity vector
data = (data - avg) / std
if data.shape[2] != 64:
    data = data[:, :, ::4, ::4]  # Downsample by a factor of 4, to (2000, 2, 64, 448)
print("Mean velocity vector:", avg)
print("Standard deviation of velocity norm:", std)

# Divide each rectangular snapshot into 7 subregions
# data_prep shape: num_subregions * time * channels * w * h
if file=='rbc_data':
    data_prep = torch.stack([data[:, :, :, k * 64:(k + 1) * 64] for k in range(7)]) #(7, 2000, 2, 64, 64); guessing this is (subregion, time, velocity component, x, y)
    print(f"data_prep shape:{data_prep.shape}")
else:
    data_prep = data.unsqueeze(0) #(1, N, c, h, w)
    print(f"data_prep shape:{data_prep.shape}")

os.makedirs(file.rsplit('.', 1)[0], exist_ok=True)
# Use sliding windows to generate samples
for j in range(0, data_prep.shape[1]):
    for i in range(data_prep.shape[0]):  #7 subregions
        # save data_prep regions separately, with array shape (time, velocity component, x, y)=(100,2,64,64)
        print((torch.FloatTensor(data_prep[i, j: j + 100]).double().float()).shape)
        torch.save(torch.FloatTensor(data_prep[i, j: j + 100]).double().float(), f"{file.rsplit('.', 1)[0]}/sample_" + str(j * data_prep.shape[0] + i) + ".pt")
        pass

print("Mean velocity vector:", avg)
print("Standard deviation of velocity norm:", std)