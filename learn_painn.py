from ase import Atoms
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData, AtomsDataModule
import pandas as pd
import numpy as np
import random
from itertools import chain
import torch
import torchmetrics
import pytorch_lightning as pl

# PaiNN (transfer) learning script for HOMO-LUMO-gaps


train_test_split_index=1 # may be 1-5
n_training_samples=209


create_db=True
if create_db:
    split=np.load("split{}.npz".format(train_test_split_index))
    train_ids = split["train_idx"]
    df = pd.read_csv("hopv.csv")
    atoms_list, property_list = [], []
    for i, row in df.iterrows():
        symbols=row["atoms"].replace("[", "").replace("]", "").replace("'", "").replace(",", "").replace(" ", "")
        coords=np.array(row["coords"].replace("[", "").replace("]", "").replace("'", "").replace(",", "").split(), dtype=float).reshape(-1,3)
        ats = Atoms(symbols, positions=coords)
        atoms_list.append(ats)
        properties = {'gap': np.array([row["gap"]])}
        property_list.append(properties)
    dataset = ASEAtomsData.create(
        './hopv.db',
        distance_unit='Ang',
        property_unit_dict={'gap': 'eV'}
    )

    ### normalization ###
    all_labels = [prop["gap"][0] for prop in property_list]
    train_labels = np.array(all_labels)[train_ids]
    train_mean = np.mean(train_labels)
    train_std = np.std(train_labels)
    for i in range(len(property_list)):
        property_list[i]["gap"] = (property_list[i]["gap"] - train_mean) / train_std

    dataset.add_systems(property_list, atoms_list)
    
    shuffled_indices = list(range(len(dataset)))
    random.Random(12345).shuffle(shuffled_indices)
    index_sublists=[shuffled_indices[int(i*len(shuffled_indices)/5):int((i+1)*len(shuffled_indices)/5)] for i in range(5)]
    for i in range(5):
        j=i+1 if i!=4 else 0
        train_indices = list(chain(*[index_sublists[k] for k in range(5) if k not in [i,j]]))[:n_training_samples]
        np.savez("split{}.npz".format(i+1), test_idx=index_sublists[i], val_idx=index_sublists[j], train_idx=train_indices)
        if i == train_test_split_index: print("Data split sizes: {} train, {} val, {} test".format(len(train_indices), len(index_sublists[j]), len(index_sublists[i])))


cutoff = 5.
n_atom_basis = 30

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)

painn = spk.representation.PaiNN(
    n_atom_basis=n_atom_basis, n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)


split=np.load("split{}.npz".format(train_test_split_index))
train_batch_size = 10 if len(split["train_idx"])%10==0 else 11
val_batch_size = 10 if len(split["val_idx"])%10==0 else 3
hopvdata = AtomsDataModule(
    './hopv.db',
    batch_size=train_batch_size,
    val_batch_size=val_batch_size,
    test_batch_size=1,
    transforms=[
        trn.ASENeighborList(cutoff=cutoff),
        trn.CastTo32()
    ],
    property_units={'gap': 'eV'},
    num_workers=1,
    split_file="split{}.npz".format(train_test_split_index),
    pin_memory=True, # set to false, when not using a GPU
    load_properties=['gap'], #only load gap property
)
hopvdata.prepare_data()
hopvdata.setup()


nnpot = spk.model.NeuralNetworkPotential(
    representation=painn,
    input_modules=[pairwise_distance],
    output_modules=[pred_gap],
    postprocessors=[trn.CastTo64()]
).to("cuda")



# to use a pre-trained model, uncomment this line and insert model path
# nnpot = torch.load("mypath/best_inference_model")



output_gap = spk.task.ModelOutput(
    name='gap',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)



task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_gap],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 5e-4},
    scheduler_cls=spk.train.ReduceLROnPlateau,
    scheduler_args={"factor": 0.5, "patience": 5},
    scheduler_monitor="val_loss"
)



logger = pl.loggers.TensorBoardLogger(save_dir="./train/")
callbacks = [
    spk.train.ModelCheckpoint(
        model_path="./train/best_inference_model",
        save_top_k=1,
        monitor="val_loss"
    )
]



trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir="./train/",
    max_epochs=10000, # for testing, we restrict the number of epochs
    log_every_n_steps=20,
)



trainer.fit(task, datamodule=hopvdata)



best_model = torch.load('./train/best_inference_model', map_location="cpu")



df = pd.read_csv("hopv.csv")
mae_total, mse_total, count = 0, 0, 0

for idx, batch in zip(split["test_idx"], hopvdata.test_dataloader()):
    gap_true = (batch['gap'].item() * train_std) + train_mean
    gap_pred = (best_model(batch)['gap'].item() * train_std) + train_mean
    mae = np.abs(gap_true-gap_pred)
    print(df.iloc[idx]["smiles"], df.iloc[idx]["gap"], gap_true, gap_pred, mae)
    mae_total+=mae
    mse_total+=mae*mae
    count+=1

print("MAE:", mae_total/count)
print("RMSE:", np.sqrt(mse_total/count))


