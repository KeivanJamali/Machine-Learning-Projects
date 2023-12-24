import data_setup, engine, model_architecture
import torch
from torch import nn
import argparse
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--datafolder", help="Please write the address of the folder which contains data.", type=str)
parser.add_argument("--city", help="Choose which city are you looking at.", type=str, choices=["luzern"])
parser.add_argument("--seed", help="What is the random you want to model on it?", type=int, default=-379)
parser.add_argument("-n", "--neural_network",
                    help="Specify hyper parameters of neural_network in order to: hidden_units, epochs, learning_rate",
                    type=str)
parser.add_argument("--early_stop", help="early_stop count?", type=int, default=0)
args = parser.parse_args()
######################################################################################################################
data_path = Path(args.datafolder)
if args.neural_network:
    hyper_parameters = args.neural_network.split(",")
city = args.city
if args.seed == -379:  # default of seed
    seed = random.choice(range(1000))
else:
    seed = args.seed
if args.early_stopping > 0:
    early_stop_count = args.early_stop
else:
    early_stop_count = None
######################################################################################################################
data = data_setup.Dataloader()
data_path = Path("D:\All Python\All_Big_raw_Data\LOS prediction\Traffic Dataset\DataLoader")
BATCH_SIZE = 32
city = "luzern"
train_dataloader, val_dataloader, test_dataloader = data.create_dataloaders(data_dir=data_path, city_code=city,
                                                                            batch_size=BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)
INPUT_SHAPE = len(train_dataloader.dataset[0][0])
HIDDEN_UNITS = int(hyper_parameters[0])
OUTPUT_SHAPE = len(data.class_names)
NUM_EPOCHS = int(hyper_parameters[1])

model1 = model_architecture.LOS_Classification_V0(in_put=INPUT_SHAPE, hidden_units=HIDDEN_UNITS, out_put=OUTPUT_SHAPE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model1.parameters(), lr=hyper_parameters[2])

model1_results = engine.train(model=model1,
                              train_dataloader=train_dataloader,
                              val_dataloader=val_dataloader,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              epochs=NUM_EPOCHS,
                              experiment_name=city,
                              model_name="Neural_Net",
                              early_stop_patience=early_stop_count,
                              device=device)
