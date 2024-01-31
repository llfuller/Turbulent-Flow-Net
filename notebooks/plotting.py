# import torch
# # import numpy as np
# # from torch.utils import data
# import matplotlib.pyplot as plt
# # import radialProfile
#
# # Load the data
# loaded_data = torch.load("../tf_seed0_bz32_inp25_pred4_lr0.001_decay0.95_coef0.0_dropout0.0_kernel3_win6pt")
# # Access individual items
# test_preds_loaded = loaded_data['test_preds']
# test_trues_loaded = loaded_data['test_trues']
# rmse_curve_loaded = loaded_data['rmse_curve']
# div_curve_loaded = loaded_data['div_curve']
# energy_spectrum_loaded = loaded_data['spectrum']
#
# print(f"rmse_curve_loaded.shape{rmse_curve_loaded.shape}")
# print(rmse_curve_loaded)
#
# fig = plt.figure()
# plt.plot(rmse_curve_loaded)
# plt.savefig("rmse_curve_plot.png", dpi=300, bbox_inches='tight')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import radialProfile
import os
# import kornia
# from torch.utils import data

a_list = [ 291.7047045,   406.81669102,  514.73942872,  615.0306772,   707.77128018,
  794.31106587,  874.04458178,  945.52944821, 1007.98369855, 1061.23767882,
 1106.48487381, 1145.32512045, 1177.99753386, 1205.92154502, 1230.2677702,
 1251.73586059, 1270.73935292, 1287.89555397, 1303.81003783, 1319.15686653,
 1334.28247318, 1349.53867414, 1365.03286025, 1381.2132499,  1398.31813472,
 1416.26246311, 1434.69870272, 1453.63135326, 1473.02121485, 1492.84819577,
 1513.01749885, 1533.38258058, 1553.6734574,  1573.4211575,  1593.12809788,
 1613.05097098, 1633.16008689, 1652.90090698, 1671.5721927,  1689.41367521,
 1706.37787875, 1722.25594206, 1736.91201401, 1750.6905879,  1763.83175885,
 1777.13668741, 1791.32775637, 1806.47865003, 1821.4914181, 1834.96029976,
 1846.95507667, 1858.30489091, 1869.87515793, 1882.32016083, 1896.37756422,
 1911.79442716, 1928.09370652, 1945.03293867, 1962.59422154, 1980.85259922]
plt.figure()
plt.plot(np.array(a_list))
plt.title("dummy plot")
plt.show()

import torch
# Load the data
# loaded_data = torch.load("../results_rbc_data_0_1000_epochs/tf_seed0_bz64_inp26_pred6_lr0.005_decay0.9_coef0.001_dropout0.0_kernel3_win6_data=rbc_data_seed=0pt")
dir = "lawson_plots_1000_epochs/tfnet_multiscale/"
# Check if the save directory exists, and create it if it does not
if not os.path.exists(dir):
 os.makedirs(dir)
# loaded_data = torch.load("..results_rbc_data/rbc_data_tfnet_multiscale_seed0_bz64_inp26_pred6_lr0.005_decay0.9_coef0.001_dropout0.0_kernel7_win10pt")
loaded_data = torch.load("../1000_epochs/results_rbc_data/rbc_data_tfnet_multiscale_seed0_bz64_inp26_pred6_lr0.005_decay0.9_coef0.001_dropout0.0_kernel7_win10pt")
# loaded_data = torch.load("../100_epochs/results_rbc_data/rbc_data_tf_seed0_bz64_inp26_pred6_lr0.005_decay0.9_coef0.001_dropout0.0_kernel3_win6pt")# Access individual items
test_preds_loaded = loaded_data['test_preds']
test_trues_loaded = loaded_data['test_trues']
rmse_curve_loaded = loaded_data['rmse_curve']
div_curve_loaded = loaded_data['div_curve']
energy_spectrum_loaded = loaded_data['spectrum']

print(f"rmse_curve_loaded.shape{rmse_curve_loaded.shape}")
print(rmse_curve_loaded)
print("here")

plt.figure()
plt.plot(np.array(rmse_curve_loaded))
plt.title("RMSE of TFNet_multiscale")
plt.xlabel("RMSE")
plt.xlabel("Prediction Step")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(dir+"rmse_curve_plot.png", dpi=300, bbox_inches='tight')
plt.show()
print("here")


def TKE(preds):
 """
 Calculate the Turbulent Kinetic Energy (TKE) for a 2D fluid dynamics simulation.

 Parameters:
 preds (numpy.ndarray): The predictions array where each row represents a different
                        instance and each column represents a spatial point in the simulation.

 Returns:
 numpy.ndarray: A 1D array of TKE values, representing the mean kinetic energy per unit mass
                due to turbulence in the flow for each instance.
 """
 mean_flow = np.expand_dims(np.mean(preds, axis=1), axis=1)
 tur_preds = np.mean((preds - mean_flow) ** 2, axis=1)
 tke = (tur_preds[0] + tur_preds[1]) / 2
 return tke


def tke2spectrum(tke):
 """
 Convert a field of Turbulent Kinetic Energy (TKE) into its spectral representation using a Fourier transform.

 Parameters:
 tke (numpy.ndarray): A 2D array representing the spatial distribution of TKE.

 Returns:
 numpy.ndarray: A 1D array representing the radial spectrum of the TKE.
 """
 """Convert TKE field to spectrum"""
 sp = np.fft.fft2(tke)
 sp = np.fft.fftshift(sp)
 sp = np.real(sp * np.conjugate(sp))
 sp1D = radialProfile.azimuthalAverage(sp)
 return sp1D


def spectrum_band(tensor):
 """
 Calculate the mean and standard deviation of the spectral bands of the TKE for a series of simulations.

 Parameters:
 tensor (numpy.ndarray): A multi-dimensional array where each element represents TKE data for a different simulation.

 Returns:
 tuple: A tuple containing the mean and standard deviation of the spectral bands.
 """
 tensor = inverse_seqs(tensor)
 spec = np.array([tke2spectrum(TKE(tensor[i])) for i in range(tensor.shape[0])])
 return np.mean(spec, axis=0), np.std(spec, axis=0)


def inverse_seqs(tensor):
 """
 Reshape and transpose a tensor to match the required input shape for TKE calculation.

 Parameters:
 tensor (numpy.ndarray): The input data tensor.

 Returns:
 numpy.ndarray: The reshaped and transposed tensor.
 """
 tensor = tensor.reshape(-1, 7, 60, 2, 64, 64)
 tensor = tensor.transpose(0, 2, 3, 1, 4, 5)
 tensor = tensor.transpose(0, 1, 2, 4, 3, 5).reshape(-1, 60, 2, 64, 448)
 tensor = tensor.transpose(0, 2, 1, 3, 4)
 return tensor


def TKE_mean(tensor):
 """
 Calculate the mean Turbulent Kinetic Energy (TKE) for a given tensor.

 Parameters:
 tensor (numpy.ndarray): A tensor representing the flow field data. It can be either already reshaped or raw.

 Returns:
 numpy.ndarray: The mean TKE value.
 """
 if tensor.shape[-1] == 448:
  return TKE(tensor)
 tensor = inverse_seqs(tensor)
 tke_mean = 0
 for i in range(0, min(70, tensor.shape[0])):
  tke_mean += TKE(tensor[i])
 tke_mean = tke_mean / tensor.shape[0]
 return tke_mean

print(f"Shape of test_preds_loaded: {test_preds_loaded.shape}") # (35, 60, 2, 64, 64) = (?,t,u or v, h,w)
# ValueError: cannot reshape array of size 491520 into shape (7,60,2,64,64)
# (t, b??, v_x or v_y, h, w) TODO: Check to see whether using only times up to 7 is valid, since plot looks slightly different
tkes = [TKE_mean(test_preds_loaded[:7])]
# title = ["Target","Con TF-net", "TF-net", "U_net",  "GAN",  "ResNet", "ConvLSTM",  "SST",  "DHPM"]
title = ["TF-net"]
fig=plt.figure(figsize=(15, 5))
# columns = 9
columns = 1
rows = 1
for i in range(columns):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(tkes[i][:64,:64])
    plt.xlabel(title[i], size = 15, rotation=0, labelpad = -100)
    plt.xticks([])
    plt.yticks([])
plt.savefig(dir+"Kinetic Energy.png", dpi = 400,bbox_inches = 'tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialization function: plot the background of each frame
def init():
    im.set_data(np.zeros((64, 64)))
    return [im]

def inverse_seqs(tensor):
 tensor = tensor.reshape(-1, 7, 60, 2, 64, 64)
 tensor = tensor.transpose(0, 2, 3, 1, 4, 5)
 tensor = tensor.transpose(0, 1, 2, 4, 3, 5).reshape(-1, 60, 2, 64, 448)
 tensor = tensor[:, :, :, :, :64]
 tensor = tensor.transpose(0, 2, 1, 3, 4)
 return tensor

animate_this_true_U = test_trues_loaded[0,:,0,:,:] #(60,64,64)
animate_this_true_V = test_trues_loaded[0,:,1,:,:]
animate_this_preds_U = test_preds_loaded[0,:,0,:,:]
animate_this_preds_V = test_preds_loaded[0,:,1,:,:]
animate_this_dict = {"truth_U":animate_this_true_U,
                     "truth_V":animate_this_true_V,
                     "preds_U":animate_this_preds_U,
                     "preds_V":animate_this_preds_V}

# Your array (35, 64, 64)
# animate_this = test_trues_loaded[:,0,0,:,:]

for data_name, data_to_animate in animate_this_dict.items():
 # Animation update function: this is called sequentially
 def update(frame):
  im.set_data(data_to_animate[frame])
  return [im]

 # Define the figure and axis for the animation
 fig, ax = plt.subplots()
 # Set up the plot
 im = ax.imshow(data_to_animate[0], cmap='viridis', interpolation='none')
 # Create the animation object
 ani = FuncAnimation(fig, update, frames=range(35), init_func=init, blit=True)
 # Save the animation to a file
 ani.save(dir+"animation_"+data_name+'.gif', writer='ffmpeg', fps=5)
 plt.close(fig)  # Close the figure to prevent it from displaying statically


# # Divergence stuff
# def divergence(preds):
#  # preds: batch_size*output_steps*2*H*W
#  preds_u = preds[:, :, 0]
#  preds_v = preds[:, :, 1]
#  u = torch.from_numpy(preds_u).float().to(device)
#  v = torch.from_numpy(preds_v).float().to(device)
#  # Sobolev gradients
#  field_grad = kornia.filters.SpatialGradient()
#  u_x = field_grad(u)[:, :, 0]
#  v_y = field_grad(v)[:, :, 1]
#  div = np.mean(np.abs((v_y + u_x).cpu().data.numpy()), axis=(0, 2, 3))
#  return div
#

fig=plt.figure()
plt.plot(div_curve_loaded, linewidth = 2, linestyle=':')
plt.title("Mean Absolute Divergence of Test Prediction", size = 18)
plt.xlabel("Prediction Step", size = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(dir+"TFNet_multiscale_divergence_test_pred", dpi=500, bbox_inches='tight')
plt.show()


fig = plt.figure()
temp_preds = test_preds_loaded[:,:,:,:,:]
spec_mean, spec_stds = spectrum_band(temp_preds)
print(f"spec_mean.shape:{spec_mean.shape}")
print(f"spec_stds.shape:{spec_stds.shape}")
x_idx = np.array(list(range(0, len(spec_mean))))
plt.plot(x_idx, spec_mean[x_idx], label=title[0], linewidth=2.5, color="k")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.yscale("log")
plt.xscale("log")

plt.legend(fontsize=14, loc=1)
plt.ylabel("Energy Spectrum", size=18)
plt.xlabel("Wave Number", size=18)
plt.ylim(10e11, )
plt.savefig(dir+"spec_ci_entire.png", dpi=400, bbox_inches='tight')