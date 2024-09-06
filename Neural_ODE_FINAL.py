import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
from tqdm import tqdm
from warnings import simplefilter
from diverse_dataloader import get_diverse_data
from model.NODE_Model import NODE_Module
from model.NN_Model import FeedForwardNN
from helpers import calc_cd, calc_area_diff, plot_area_diff, plot_sample_training, add_gaussian
from helpers_folder import create_folder
from dataloader import get_data
from sys import exit
from argparse import ArgumentParser

# Seed the random number generator
torch.manual_seed(0)
np.random.seed(0)

parser = ArgumentParser(description="Waterline Prediction")
parser.add_argument('-e', '--epochs', default=2000, help="Number of epochs.", type=int)
parser.add_argument('-m', "--mode", default="NODE", help="Use NODE or NN.", type=str)
parser.add_argument('-s', "--solver", default="dopri5", help="NODE-Solver, use dopri5, euler. Check repo for more solver.", type=str)
parser.add_argument('-l', '--hidden_layers', default=15, help="Number of hidden layers.", type=int)
parser.add_argument('-n', '--neurons', default=100, help="Number of neurons.", type=int)
parser.add_argument('-z', '--z_dim', default=8, help="Size of encoder output.", type=int)
parser.add_argument('-a', "--activation", default="elu", help="Activation Function.", type=str)
parser.add_argument('-t', "--dataset", default="random", help="Input: random, first or last. Train/Test Random, FlowProp-First or FlowProp-Last", type=str)
parser.add_argument('-d', '--diverse', default=False, help="Test FlowProp-Diverse", type=bool)
parser.add_argument('-g', '--gaussian', default=False, help="Add gaussian noise", type=bool)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_folder = f"NODE_{args.hidden_layers}"

# if random_split == False and use_first == True -> FlowProp-First // if random_split == False and use_first == False -> FlowProp-Last
if args.dataset.lower() == "random":
    train_data, validation_data, shuffle_idx_train, shuffle_idx_val = get_data(samples_train=-1, samples_test=-1,
                                                                               split=0.8, random_split=True,
                                                                               use_first=False)

elif args.dataset.lower() == "first":
    train_data, validation_data, shuffle_idx_train, shuffle_idx_val = get_data(samples_train=-1, samples_test=-1,
                                                                               split=0.8, random_split=False,
                                                                               use_first=True)
    main_folder += "_first"

elif args.dataset.lower() == "last":
    train_data, validation_data, shuffle_idx_train, shuffle_idx_val = get_data(samples_train=-1, samples_test=-1,
                                                                               split=0.8, random_split=False,
                                                                               use_first=False)
    main_folder += "_last"

else:
    print("Must be random, first or last.")
    exit(0)


plot_path = create_folder(main_folder, "final_training")
plot_test_path = create_folder(main_folder,"test_samples")
plot_diverse_path = create_folder(main_folder, "diverse_samples")
model_path = create_folder(main_folder, "model")

if args.mode.upper() == "NODE":
    model = NODE_Module(dim_x=args.z_dim, dim_z=1, hidden_size=args.neurons, n_layers_hidden=args.hidden_layers, activation=args.activation, z_dim=args.z_dim, solver=args.solver).to(device)
elif args.mode.upper() == "NN":
    model = FeedForwardNN(encoded_dim=args.z_dim, hidden_dim=args.neurons, output_dim=1, n_layers_hidden=args.hidden_layers, activation=args.activation).to(device)
else:
    print("Error model not provided or implemented. Please use NODE or NN.")
    exit(0)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of trainable parameters: ", params)

################### Main Training ############################
# Define the loss function, optimizer, scheduler
loss_fn = nn.MSELoss()
train_loss = []
validation_loss = []
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

#final training
#data preparation
y_0 = train_data[1][:, 0, :].to(device)
s_train = train_data[0].squeeze(-1).to(device)
y_train = train_data[1].to(device)
x_train = train_data[2].to(device)

y_0_val = validation_data[1][:, 0, :].to(device)
s_val = validation_data[0].squeeze(-1).to(device)
y_val = validation_data[1].to(device)
x_val = validation_data[2].to(device)

n_epochs = args.epochs

best_loss = np.inf

for epoch in tqdm(range(1, n_epochs+1)):
    optimizer.zero_grad()
    if args.mode.upper() == "NODE":
        y_pred = model(s_train, y_0, x_train)
        loss = loss_fn(y_pred, y_train)
    elif args.mode.upper() == "NN":
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train[:, 120])
    else:
        print("Error model not provided or implemented. Please use NODE or NN.")
        break
    loss.backward()
    optimizer.step()
    scheduler.step()

    #validation data
    with torch.no_grad():
        if args.mode.upper() == "NODE":
            if not args.gaussian:
                y_pred_test = model(s_val, y_0_val, x_val)
            elif args.gaussian:
                y_pred_test = model(s_val, y_0_val, add_gaussian(x_val, device=device, mean=0, std=0.05))  # 5 % noise
            loss_val = loss_fn(y_pred_test, y_val)
        elif args.mode.upper() == "NN":
            y_pred_test = model(x_val)
            loss_val = loss_fn(y_pred_test, y_val[:, 120])
        else:
            print("Error model not provided or implemented. Please use NODE or NN.")
            exit(0)
        print(f"Epoch {epoch}/{n_epochs} Loss Train: {loss.item()} Loss Val: {loss_val.item()}")
        train_loss.append(loss.item())
        validation_loss.append(loss_val.item())

    if loss.item() < 1e-8:
        break
    if epoch % 10 == 0 or epoch == 1:
        path = Path(plot_path, f'epoch_{epoch}.png')
        plot_sample_training(train_data[1], y_pred, path, sample_nr=0, title=f'Epoch {epoch}/{n_epochs} Loss: {loss.item()}')

    if loss_val.item() < best_loss:
        best_loss = loss_val.item()
        print("Best model at epoch: %i, with loss: %.8f" % (epoch, best_loss))

        PATH = f"{model_path}/NODE_{args.hidden_layers}.tar"

        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss":best_loss,
                    },
                   PATH)


#plot the loss of training using matplotlib
plt.plot(train_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(Path(plot_path.parent, 'losses.png'))
plt.close()

if args.diverse:
    diverse_data, idx_diverse = get_diverse_data()
    y_0_diverse = diverse_data[1][:, 0, :].to(device)
    s_diverse = diverse_data[0].squeeze(-1).to(device)
    y_diverse = diverse_data[1].to(device)
    x_diverse = diverse_data[2].to(device)

    with torch.no_grad():
        if args.mode.upper() == "NODE":
            if not args.gaussian:
                y_pred_diverse = model(s_diverse, y_0_diverse, x_diverse)
            elif args.gaussian:
                y_pred_diverse = model(s_diverse, y_0_diverse, add_gaussian(x_diverse, device=device, mean=0, std=0.05))  # 5% noise
            loss_val = loss_fn(y_pred_diverse, y_diverse)
            s_diverse = diverse_data[0].cpu().detach().numpy()
            y_diverse = diverse_data[1].cpu().detach().numpy()
            h0 = y_diverse[:, 120, :]  # 3 * P
            cd = calc_cd(h0)

            y_pred_diverse = y_pred_diverse.cpu().detach().numpy()
            h0_pred = y_pred_diverse[:, 120, :]  # 3 * P
            cd_pred = calc_cd(h0_pred)

            print(5 * "-" + "Diverse Weirs Data" + 5 * "-")
            print("MSE: ", np.mean((y_pred_diverse - y_diverse) ** 2))
            print("MSE for cd: ", np.mean((cd_pred - cd) ** 2))

            sample_names = idx_diverse

            area_hat = 0

            for i in range(y_diverse.shape[0]):
                file_path = f"{plot_diverse_path}/Sample_{sample_names[i]}.png"
                plot_area_diff(y_diverse[i, :, :].reshape(-1), y_pred_diverse[i, :, :].reshape(-1), file_path=file_path)
                area_hat += calc_area_diff(y_diverse[i, :, :].reshape(-1), y_pred_diverse[i, :, :].reshape(-1))

            print("MAE in m**2 of AUC: ", area_hat / y_val.shape[0])
        elif args.mode.upper() == "NN":
            y_pred_diverse = model(x_diverse)
            loss_val = loss_fn(y_pred_diverse, y_diverse[:, 120])

            h0 = y_diverse[:, 120, :].cpu().detach().numpy()  # 3 * P
            cd = calc_cd(h0)

            h0_pred = y_pred_diverse.cpu().detach().numpy()
            cd_pred = calc_cd(h0_pred)

            print(5*"-" + "Diverse Weirs Data" + 5*"-")
            print("MSE for cd: ", np.mean((cd_pred - cd)**2))
        else:
            print("Error model not provided or implemented. Please use NODE or NN.")
            exit(0)

if args.mode.upper() == "NODE":
    #plot everything with plotly
    #plot the training data
    #tensors to dataframes
    s_train = train_data[0].cpu().detach().numpy()
    y_train = train_data[1].cpu().detach().numpy()
    h0 = y_train[:, 120, :] # 3 * P
    cd = calc_cd(h0)

    y_pred = y_pred.cpu().detach().numpy()
    h0_pred = y_pred[:, 120, :] # 3 * P
    cd_pred = calc_cd(h0_pred)

    print(5*"-" + "Train Data" + 5*"-")

    print("MSE:", np.mean((y_pred - y_train) ** 2))
    print("MSE for cd: ", np.mean((cd_pred - cd)**2))

    area_hat = 0
    for i in range(y_train.shape[0]):
        area_hat += calc_area_diff(y_train[i, :, :].reshape(-1), y_pred[i, :, :].reshape(-1))
    print("MAE in m**2 of AUC: ", area_hat/y_train.shape[0])

    sample_names = shuffle_idx_train

    # supress warnings by pd
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    df_results = pd.DataFrame(s_train[0], columns=['Distance'])
    for i in range(y_train.shape[0]):
        df_results[f'Sample_{sample_names[i]}_Ground_Truth'] = y_train[i]
        df_results[f'Sample_{sample_names[i]}_Predictions'] = y_pred[i]

    fig = px.line(df_results, x='Distance', y=df_results.columns[1:], title='Training Data')

    #save the plot as html
    pio.write_html(fig, str(Path(plot_path.parent,'training_data.html')))

    #plot the validation data
    #tensors to dataframes
    s_val = validation_data[0].cpu().detach().numpy()
    y_val = validation_data[1].cpu().detach().numpy()
    h0 = y_val[:, 120, :] # 3 * P
    cd = calc_cd(h0)

    y_pred_val = y_pred_test.cpu().detach().numpy()
    h0_pred = y_pred_val[:, 120, :] # 3 * P
    cd_pred = calc_cd(h0_pred)

    print(5*"-" + "Validation Data" + 5*"-")

    print("MSE: ", np.mean((y_pred_val - y_val)**2))
    print("MSE for cd: ", np.mean((cd_pred - cd)**2))

    sample_names = shuffle_idx_val

    area_hat = 0
    for i in range(y_val.shape[0]):
        file_path = f"{plot_test_path}/Sample_{sample_names[i]}.png"
        plot_area_diff(y_val[i, :, :].reshape(-1), y_pred_val[i, :, :].reshape(-1), file_path=file_path)
        area_hat += calc_area_diff(y_val[i, :, :].reshape(-1), y_pred_val[i, :, :].reshape(-1))

    print("MAE in m**2 of AUC: ", area_hat/y_val.shape[0])

    #create a dataframe and name the columns according to the sample names
    df_results = pd.DataFrame(s_val[0], columns=['Distance'])

    for i in range(y_val.shape[0]):
        df_results[f'Sample_{sample_names[i]}_Ground_Truth'] = y_val[i]
        df_results[f'Sample_{sample_names[i]}_Predictions'] = y_pred_val[i]

    fig = px.line(df_results, x='Distance', y=df_results.columns[1:], title='Validation Data')
    #save the plot as html
    pio.write_html(fig, str(Path(plot_path.parent, 'validation_data.html')))

    #plot the losses of training and pretraining
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(train_loss)), y=train_loss, mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(x=np.arange(len(validation_loss)), y=validation_loss, mode='lines', name='Validation Loss'))
    fig.update_layout(title='Losses', xaxis_title='Epoch', yaxis_title='Loss')
    #save the plot as html
    pio.write_html(fig, str(Path(plot_path.parent, 'losses.html')))

elif args.mode.upper() == "NN":
    h0 = y_val[:, 120, :].cpu().detach().numpy()  # 3 * P
    cd = calc_cd(h0)
    y_pred_val = y_pred_test.cpu().detach().numpy()
    cd_pred = calc_cd(y_pred_val)
    print(5 * "-" + "Validation Data" + 5 * "-")
    print("MSE for cd: ", np.mean((cd_pred - cd) ** 2))

else:
    print("Error model not provided or implemented. Please use NODE or NN.")
    exit(0)






