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
plt.show()

import torch

# Load the data
loaded_data = torch.load("../tf_seed0_bz32_inp25_pred4_lr0.001_decay0.95_coef0.0_dropout0.0_kernel3_win6pt")
# Access individual items
test_preds_loaded = loaded_data['test_preds']
test_trues_loaded = loaded_data['test_trues']
rmse_curve_loaded = loaded_data['rmse_curve']
div_curve_loaded = loaded_data['div_curve']
energy_spectrum_loaded = loaded_data['spectrum']

print(f"rmse_curve_loaded.shape{rmse_curve_loaded.shape}")
print(rmse_curve_loaded)

plt.figure()
plt.plot(np.array(rmse_curve_loaded))
plt.title("RMSE of TFNet")
plt.xlabel("RMSE")
plt.xlabel("Prediction Step")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("lawson_plots/rmse_curve_plot.png", dpi=300, bbox_inches='tight')
plt.show()