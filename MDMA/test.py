##################################################
# example script to train the multclassification #
##################################################

import torch
import torch.optim as optim

from dataloader import ParticleCloudDataloader
from discriminator import Disc
def accuracy(pred_logits, labels):
    # Get the index of the max log-probability (predicted class) for each input in the batch
    pred_labels = torch.argmax(pred_logits, dim=1)

    # Check how many of the predictions are correct by comparing with the true labels
    correct_preds = pred_labels == labels

    # Calculate the accuracy as the average number of correct predictions
    acc = correct_preds.float().mean()
    return acc
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# Choose appropriate loss function (criterion) and optimizer
criterion = torch.nn.CrossEntropyLoss()

datamodule = ParticleCloudDataloader("treeGan_dataset_2", 128, in_dir="/beegfs/desy/user/kaechben/testing/")
datamodule.setup("train")
dataloader = datamodule.train_dl
datamodule_fake = ParticleCloudDataloader("MDMA_dataset_2", 128, in_dir="/beegfs/desy/user/kaechben/testing/")
datamodule_fake.setup("train")
dataloader_fake = datamodule_fake.train_dl
datamodule_test = ParticleCloudDataloader("groundtruth_dataset_2", 128, in_dir="/beegfs/desy/user/kaechben/testing/")
datamodule_test.setup("train")
dataloader_test = datamodule_test.train_dl

model = Disc(n_dim=4, latent_dim=16, hidden_dim=64, out_classes=2, num_layers=5, heads=16, dropout=0.2, avg_n=1601)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses={"real":[],"fake":[]}
acc={"real":[],"fake":[]}
for epoch in range(100):  # loop over the dataset multiple times, adjust epochs as needed


    if epoch>0:
        print("TRAINING")
        print("loss: ", torch.tensor(losses["fake"]).mean()/2+torch.tensor(losses["real"]).mean()/2)
        print("acc: ",torch.tensor(acc["fake"]).mean()/2+torch.tensor(acc["real"]).mean()/2)
        losses={"real":[],"fake":[]}
        acc={"real":[],"fake":[]}
        model.eval()
        with torch.no_grad():
            for (_,batch),(_,fake) in zip(enumerate(datamodule.val_dl,0),enumerate(datamodule_fake.val_dl,0)):
                inputs, r_label, r_mask,r_Einc = batch[0].to(device), torch.ones_like(batch[0][:, 0, 1]).to(device).long(), batch[1].to(device), batch[2].to(device).squeeze(-1)
                fakes, f_label, f_mask,f_Einc = fake[0].to(device), torch.zeros_like(fake[0][:, 0, 1]).to(device).long(), fake[1].to(device), fake[2].to(device).squeeze(-1)

                outputs = model(inputs, r_mask,r_Einc)
                loss_real = criterion(outputs, r_label)
                acc_real = accuracy(outputs,r_label)
                outputs = model(fakes, f_mask,f_Einc)
                loss_fake = criterion(outputs, f_label)
                acc_fake = accuracy(outputs,f_label)

                losses["real"].append(loss_real)
                losses["fake"].append(loss_fake)
                acc["real"].append(acc_real)
                acc["fake"].append(acc_fake)

            print("VALIDATION")
            print("loss: ",torch.tensor(losses["fake"]).mean()/2+torch.tensor(losses["real"]).mean()/2)
            print("acc: ",torch.tensor(acc["fake"]).mean()/2+torch.tensor(acc["real"]).mean()/2)
            losses={"real":[],"fake":[]}
            acc={"real":[],"fake":[]}
        model.train()
    for (i, data), (j, fake) in zip(enumerate(dataloader, 0), enumerate(dataloader_fake, 0)):
        # get the inputs; data is a list of [inputs, mask (True for padded particles), labels)]
        inputs, r_label, r_mask, r_Einc = data[0].to(device), torch.ones_like(data[0][:, 0, 1]).to(device).long(), data[1].to(device), data[2].to(device).squeeze(-1)
        fakes, f_label, f_mask, f_Einc = fake[0].to(device), torch.zeros_like(fake[0][:, 0, 1]).to(device).long(), fake[1].to(device), fake[2].to(device).squeeze(-1)
        # zero the parameter gradients
        if fakes.isnan().any() or inputs.isnan().any():
            print("nans")
            continue
        optimizer.zero_grad()
        # forward + backward + optimize + logging
        outputs = model(inputs, r_mask,r_Einc)
        loss_real = criterion(outputs, r_label)
        acc_real = accuracy(outputs,r_label)
        outputs = model(fakes, f_mask,f_Einc)
        loss_fake = criterion(outputs, f_label)
        acc_fake = accuracy(outputs,f_label)

        losses["real"].append(loss_real.item())
        losses["fake"].append(loss_fake.item())
        acc["real"].append(acc_real.item())
        acc["fake"].append(acc_fake.item())

        loss=(loss_real+loss_fake)/2
        loss.backward()
        optimizer.step()
    if epoch>3:
        acc={"real":[]}
        with torch.no_grad():
            for (i, data) in enumerate(dataloader_test,0):
                inputs, r_label, r_mask, r_Einc = data[0].to(device), torch.ones_like(data[0][:, 0, 1]).to(device).long(), data[1].to(device), data[2].to(device).squeeze(-1)
                outputs = model(inputs, r_mask,r_Einc)
                pred_labels = torch.argmax(outputs, dim=1)
                acc["real"].append(pred_labels.cpu())
            print("acc: ",torch.cat(acc["real"]).float().mean())
        break

print("Finished Training")
