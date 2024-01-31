import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.utils import data
import time
import random
from models import TFNet, DivergenceLoss, TFNet_multiscale
from models import penalty
from tqdm import trange
# from models.baselines import ConvLSTM, DHPM, GAN, ResNet, SST, Unet  # Assuming the class is named Unet
from models.baselines import DHPM, ResNet, Unet  # Assuming the class is named Unet
from models.baselines import ConvLSTM
from models.baselines import ResNetMini
from models.baselines import GAN
import os
# import models.baselines.DHPM
from functools import partial
from neuralop.models import FNO
import os


from utils import Dataset, train_epoch, eval_epoch, test_epoch, divergence, spectrum_band, update_train_util_device
from multiprocessing import freeze_support

if __name__ == '__main__':

    def reshape_data_for_convLSTM(data):
        # data is of shape (b, 2*t, h, w)
        b, two_t, h, w = data.shape
        t = two_t // 2  # Assuming 2*t is always even
        
        # Split the data into two halves along the second axis (time)
        data_height, data_width = data.chunk(2, dim=1)
        
        # Stack the halves to create a new channel dimension
        # Now we have shape (b, 2, t, h, w), where 2 is for height and width
        data_stacked = torch.stack((data_height, data_width), dim=2)
        
        # Transpose to get (t, b, c, h, w)
        data_reshaped = data_stacked.permute(2, 0, 1, 3, 4)
        
        return data_reshaped

    freeze_support()  # Needed when freezing to create a standalone executable

    parser = argparse.ArgumentParser(description='Tuburlent-Flow Nets')
    parser.add_argument('--max_kernel_size', type=int, required=False, default="3", help='max convolution kernel size')
    parser.add_argument('--max_time_range', type=int, required=False, default="6", help='max moving average window size for temporal filter')
    parser.add_argument('--output_length', type=int, required=False, default="4", help='number of prediction losses used for backpropagation')
    parser.add_argument('--input_length', type=int, required=False, default="26", help='input length')
    parser.add_argument('--batch_size', type=int, required=False, default="32", help='batch size')
    parser.add_argument('--num_epoch', type=int, required=False, default="50", help='maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, required=False, default="0.0001", help='learning rate')
    parser.add_argument('--decay_rate', type=float, required=False, default="0.95", help='learning decay rate')
    parser.add_argument('--dropout_rate', type=float, required=False, default="0.0", help='dropout rate')
    parser.add_argument('--coef', type=float, required=False, default="0.0", help='the coefficient for divergence free regularizer')
    parser.add_argument('--inp_dim', type=int, required=False, default="2", help='number of channels per frames')
    parser.add_argument('--seed', type=int, required=False, default="0", help='random seed')
    parser.add_argument('--d_id', type=int, required=False, default="0", help='device id')
    parser.add_argument('--model', type=str, required=False, default="tf", help='model to run')
    parser.add_argument('--data', type=str, required=False, default="rbc_data", help='data to run')
    
    args = parser.parse_args()

    train_direc = f"{args.data}/sample_"
    test_direc = f"{args.data}/sample_"
    save_direc = f"1000_epochs/results_{args.data}/"

    device=torch.device(f"cuda:{args.d_id}" if torch.cuda.is_available() else "cpu")
    update_train_util_device(device)
    DHPM.update_device(device)
    ResNetMini.update_device(device)
    penalty.update_device(device)

    # plt.plot(idx+1, tf_con['loss_curve'][idx]*stds, label = "Con TF-net", marker=markers[1], linewidth = 1.5, color = colors[1])
    # plt.plot(idx+1,  tf['loss_curve'][idx]*stds, label = "TF-net", marker=markers[2], linewidth = 3, color = colors[2])
    # plt.plot(idx+1, u['loss_curve'][idx]*stds, label = "U_net", marker=markers[3], linewidth = 1.5, color = colors[3])
    # plt.plot(idx+1, gan['loss_curve'][idx]*stds, label = "GAN", marker=markers[4], linewidth = 1.5, color = colors[4])
    # plt.plot(idx+1, cnn['loss_curve'][idx]*stds, label = "ResNet", marker= markers[5], linewidth = 1.5, color = colors[5])
    # plt.plot(idx+1, convlstm['loss_curve'][idx]*stds, label = "ConvLSTM", marker= markers[6], linewidth = 1.5, color = colors[6])
    # plt.plot(idx+1, sst['loss_curve'][idx]*stds, label = "SST", marker= markers[7], linewidth = 1.5, color = colors[7])
    # plt.plot(idx+1, dhpm['loss_curve'][idx]*stds, label = "DHPM", marker=markers[8], linewidth = 1.5, color = colors[8])
    model_dict = {"tfnet_multiscale":TFNet_multiscale,
                #   "tf":TFNet,
                #   "u":Unet.U_net, 
                #   "gan":[Unet.U_net, GAN.Discriminator_Spatial],  
                #   "convlstm":ConvLSTM.CLSTM, 
                #   "fno": FNO,
                #   "sst":None, 
                #   "dhpm":DHPM.DHPM, 
                # "resnet":ResNet.ResNet,
                # "resnetmini":ResNetMini.ResNet
                }
    model_dict = {args.model: model_dict[args.model]}

    for model_str, model_class in model_dict.items():
        print(f"Model Name:{model_str}")
        print(f"Model Class:{model_class}")

        random.seed(args.seed)  # python random generator
        np.random.seed(args.seed)  # numpy random generator

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        max_time_range = args.max_time_range
        output_length = args.output_length
        input_length = args.input_length
        learning_rate = args.learning_rate
        dropout_rate = args.dropout_rate
        max_kernel_size = args.max_kernel_size
        batch_size = args.batch_size
        num_epoch = args.num_epoch
        coef = args.coef
        decay_rate = args.decay_rate
        inp_dim = args.inp_dim

        os.makedirs("assets_" + args.data, exist_ok=True)
        model_name = "{}_{}_seed{}_bz{}_inp{}_pred{}_lr{}_decay{}_coef{}_dropout{}_kernel{}_win{}".format(args.data, 
                                                                                                        model_str,
                                                                                                        args.seed,
                                                                                                        batch_size,
                                                                                                        input_length,
                                                                                                        output_length,
                                                                                                        learning_rate,
                                                                                                        decay_rate,
                                                                                                        coef,
                                                                                                        dropout_rate,
                                                                                                        max_kernel_size,
                                                                                                        max_time_range)


        # train-valid-test split
        train_indices = list(range(0, 6000))
        valid_indices = list(range(6000, 7700))
        test_indices = list(range(7700, 9800))


        train_set = Dataset(train_indices, input_length + max_time_range - 1, 40, output_length, train_direc, True)
        valid_set = Dataset(valid_indices, input_length + max_time_range - 1, 40, 6, test_direc, True)
        train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
        valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 2)

        # Load one sample from the training set
        # print(f"train_set.shape:{train_set[0].shape}")
        sample_x, sample_y = train_set[0]
        things_in_train_set = sample_x.shape[0]
        print(len(train_set))
        print(f"sample_x.shape:{sample_x.shape}")
        print(f"sample_y.shape:{sample_y.shape}")
        print(f"sample_x[0].shape:{sample_x[0].shape}")

        # if model_str == 'tf': # runs fine on home PC, RMSE greater than expected
        #     model = TFNet(input_channels = input_length*inp_dim,
        #                 output_channels = inp_dim,
        #                 kernel_size = max_kernel_size,
        #                 dropout_rate = dropout_rate,
        #                 time_range = max_time_range).to(device)
        if model_str == 'tfnet_multiscale': # runs fine on home PC, RMSE greater than expected
            model = TFNet_multiscale.TFNet_multiscale(input_channels = input_length*inp_dim,
                        output_channels = inp_dim,
                        max_kernel_size = max_kernel_size,
                        dropout_rate = dropout_rate,
                        max_time_range = max_time_range).to(device)

        # if model_str == 'fno': # runs fine on home PC, RMSE greater than expected
        #     model = model_class(n_modes=(64, 64), hidden_channels=64,
        #         in_channels=(input_length + max_time_range - 1)*inp_dim, out_channels=2).to(device)
        # elif model_str == 'u':
        #     model = model_class(input_channels = (input_length + max_time_range - 1)*inp_dim,
        #                         output_channels = inp_dim,
        #                         kernel_size = max_kernel_size,
        #                         dropout_rate = dropout_rate).to(device)
        # elif model_str == 'resnet': # runs on home  PC, keep batch size <= 8
        #     model = ResNetMini.ResNet(input_channels = (input_length + max_time_range - 1)*inp_dim,#input_length*inp_dim,#input_length*inp_dim,
        #                         output_channels = inp_dim,
        #                         kernel_size = max_kernel_size).to(device)
        # elif model_str == 'resnetmini': # runs on home  PC, keep batch size <= 8
        #     model = ResNetMini.ResNet(input_channels = 62,#input_length*inp_dim,#input_length*inp_dim,
        #                         output_channels = inp_dim,
        #                         kernel_size = max_kernel_size).to(device)
        #                         # python run_model.py --time_range=6 --output_length=6 --input_length=26 --batch_size=2 --learning_rate=0.005 --decay_rate=0.9 --coef=0.001 --seed=0 --model=resnetmini

        # elif model_str == 'convlstm':
        #     model = ConvLSTM.CLSTM(input_size = sample_x[0].shape,#input_length*inp_dim,#input_length*inp_dim
        #                            ).to(device)

        # elif model_str == 'dhpm':
        #     model = model_class(hidden_dim = [200,200], 
        #                         num_layers = [3,3]).to(device)
        # elif model_str == 'gan':
        #     model = model_class[0](input_channels = (input_length + max_time_range - 1)*inp_dim,
        #                         output_channels = inp_dim,
        #                         kernel_size = max_kernel_size,
        #                         dropout_rate = dropout_rate).to(device)
        #     discriminator = model_class[1](input_channels = inp_dim).to(device).train()
        #     disc_optimizer = torch.optim.Adam(discriminator.parameters(), learning_rate)
        #     disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size = 1, gamma = decay_rate)
        #     bce_loss = nn.BCELoss()
        #     mse_loss = torch.nn.MSELoss()
        #     d_loss = torch.tensor([0], requires_grad=True, dtype=torch.float32).to(device)
        #     d_loss_cum = []
        #     current_epoch = 0
        #     warmup=0

        #     def gan_loss(preds, trues):
        #         if current_epoch < warmup:
        #             return mse_loss(preds, trues)
        #         global bce_loss, discriminator, d_loss, d_loss_cum
        #         real = torch.autograd.Variable(torch.Tensor(preds.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        #         fake = torch.autograd.Variable(torch.Tensor(preds.shape[0], 1).fill_(0.0), requires_grad=False).to(device)

        #         g_loss = bce_loss(discriminator(preds), real)

        #         real_loss = bce_loss(discriminator(trues), real)
        #         fake_loss = bce_loss(discriminator(preds.detach()), fake)

        #         d_loss += (real_loss + fake_loss) / 2
        #         d_loss_cum.append(d_loss.item() / args.output_length)
        #         return g_loss
            
        #     def update_disc():
        #         global d_loss, disc_optimizer
        #         disc_optimizer.zero_grad()
        #         d_loss.backward()
        #         disc_optimizer.step()
        #         d_loss=torch.tensor([0], requires_grad=True, dtype=torch.float32).to(device)
                
        else:
            print("No Match.")

        # model_ResNet = ResNet.ResNet(input_channels = input_length*inp_dim,
        #             output_channels = inp_dim,
        #             kernel_size = kernel_size).to(device)
        # print("TF")
        # print(model)
        # print("RESNET")
        # print(model)
        # elif model_name == 'convlstm':
        #     model = model_class(input_size=(64, 64), 
        #                         channels = inp_dim, 
        #                         hidden_dim = [64], 
        #                         num_layers = 1).to(device)

        # for batch_data, batch_labels in train_loader:
        #     print(f"batch size:{batch_data.size()}")

        train_loss_fun = torch.nn.MSELoss() if not model_str == 'gan' else gan_loss
        val_loss_fun = torch.nn.MSELoss()
        regularizer = DivergenceLoss(torch.nn.MSELoss())
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = decay_rate)

        # print("Initial eval results", eval_epoch(valid_loader, model, val_loss_fun)[0])

        train_rmse = []
        valid_rmse = []
        test_rmse = []
        min_rmse = 1e8

        # Assuming 'model' is your PyTorch model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Number of trainable parameters: {num_params}")
        print(f"num_epochs:{num_epoch}")
        for i in trange(num_epoch):
            start = time.time()
            torch.cuda.empty_cache()
            # Iterate through the batches in train_loader
            # for i, (x_batch, y_batch) in enumerate(train_loader):
            #     print("Shape of x_batch:", x_batch.shape)
            #     print("Shape of y_batch:", y_batch.shape)
            # print("Does train_loader have a shape?")
            # print(f"train_loader.shape:{train_loader.shape}")
            model.train()


            train_rmse.append(train_epoch(train_loader, model, optimizer, train_loss_fun, coef, regularizer, update_disc if model_str == 'gan' else None))

            model.eval()
            rmse, preds, trues = eval_epoch(valid_loader, model, val_loss_fun)
            valid_rmse.append(rmse)

            if valid_rmse[-1] < min_rmse:
                min_rmse = valid_rmse[-1]
                best_model = model
                torch.save({
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_rmse': train_rmse,
                    'valid_rmse': valid_rmse,
                    # any other states or tensors you want to save
                }, f"{save_direc}{model_name}_complete_model.pth")
            end = time.time()

            # Early stopping
            # if (len(train_rmse) > 100 and np.mean(valid_rmse[-5:]) >= np.mean(valid_rmse[-10:-5])):
            #         break

            print("Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}".format(i+1, (end-start)/60, train_rmse[-1], valid_rmse[-1]))
            scheduler.step()
            if model_str == 'gan':
                print(" Disc loss: {:0.3f}".format(round(np.sqrt(np.mean(d_loss_cum)), 5)))
                d_loss_cum = []
                current_epoch=i
                disc_scheduler.step()

        if 'best_model' not in locals() and 'best_model' not in globals():
            best_model = model
            best_model.load_state_dict(torch.load(f"{model_name}_complete_model.pth")['model_state_dict'])

        # Testing
        torch.cuda.empty_cache()
        test_set = Dataset(test_indices, input_length + max_time_range - 1, 40, 60, test_direc, True)
        test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 2)
        test_preds, test_trues, rmse_curve = test_epoch(test_loader, best_model, val_loss_fun)

        # Denormalization: Optional
        if args.data == 'rbc_data':
            mean_vec = np.array([-1.6010, 0.0046]).reshape(1, 1, 2, 1, 1)
            norm_std = 2321.9727
        elif args.data == 'data9_101':
            mean_vec = np.array([0.0, 0.0]).reshape(1, 1, 2, 1, 1)
            norm_std = 1.8061
        elif args.data == 'data21_101':
            mean_vec = np.array([0.0, 0.0]).reshape(1, 1, 2, 1, 1)
            norm_std = 2.7431
        elif args.data == 'data20_101':
            mean_vec = np.array([0.0, 0.0]).reshape(1, 1, 2, 1, 1)
            norm_std = 2.5912
        else:
            raise ValueError("No such dataset")
        test_preds = test_preds * norm_std + mean_vec
        test_trues = test_trues * norm_std + mean_vec


        # Compute evaluation scores
        rmse_curve = np.sqrt(np.mean((test_preds - test_trues)**2, axis = (0,2,3,4)))
        div_curve = divergence(test_preds)
        energy_spectrum = spectrum_band(test_preds)

        # Check if the save directory exists, and create it if it does not
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        torch.save({"test_preds": test_preds[::60],
                    "test_trues": test_trues[::60],
                    "rmse_curve": rmse_curve,
                    "div_curve": div_curve,
                    "spectrum": energy_spectrum},
                    save_direc+model_name +"pt")
                
        # torch.save({"test_preds": test_preds[::60],
        #     "test_trues": test_trues[::60],
        #     "rmse_curve": rmse_curve, 
        #     "div_curve": div_curve, 
        #     "spectrum": energy_spectrum}, 
        #     model_name + "pt")

        torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_rmse': train_rmse,
            'valid_rmse': valid_rmse,
            # any other states or tensors you want to save
        }, f"{save_direc}{model_name}_complete_model.pth")

        np.save(f"{save_direc}{model_name}_loss_curve.npy", rmse_curve)


        # save command line args used
        arg_string = ""
        for arg in vars(args):
            value = getattr(args, arg)
            arg_string += f" --{arg} {value}"
        with open(save_direc+f"tfnet_multiscale_terminal_commands_data{args.data}_seed{str(args.seed)}.txt", 'w') as file:
            file.write(arg_string)