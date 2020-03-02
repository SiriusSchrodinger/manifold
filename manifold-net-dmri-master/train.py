import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from tensorboardX import SummaryWriter

import time

from model import ManifoldNetSPD, ParkinsonsDataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


training_params = {'batch_size': 30,
          'shuffle': True,
          'num_workers': 0}

validation_params = {'batch_size': 71,
        'shuffle': False,
        'num_workers': 0}

max_epochs = 150
validate = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

manifold_net_con = ManifoldNetSPD().to(device)
print('Parameters:')
print(count_parameters(manifold_net_con))
dataset = ParkinsonsDataset('../data.npz')
training, validation = data.random_split(dataset, [dataset.__len__()-71,71])

training_generator = data.DataLoader(training, **training_params)
validation_generator = data.DataLoader(validation, **validation_params)

optimizer_con = optim.Adam(manifold_net_con.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss().cuda()

def classification(out,desired):
    _, predicted = torch.max(out, 1)
    total = desired.shape[0]
    correct = (predicted == desired).sum().item()
    return correct/float(total)

def my_loss(sample, out):
    diff = sample - out
    result = torch.norm(diff)
    return result


writer = SummaryWriter()
i = 0
# Training...
print('Starting training...')
with open("data_backup", "w") as f:
    try:
        for epoch in range(max_epochs):
            print('Starting Epoch ', epoch, '...')
            #for sample, label in training_generator:
            epoch_start = time.time()
            for sample in training_generator:
                batch_start = time.time()
                i += 1
                sample = sample.to(device)
                #label = label.to(device)
                
                #print("\n")
                start = time.time()
                optimizer_con.zero_grad()
                #import pdb; pdb.set_trace()
                out = manifold_net_con(sample)
                #print(out)
                #print(label)
                #loss = criterion(out,label)
                loss = my_loss(sample, out)
                #backward_start = time.time()
                loss.backward()
                #backward_end = time.time()
                #print("backward time:", backward_end - backward_start)
                end = time.time()
                #print("\n")

                optimizer_con.step()
                #print('Training Loss: ', loss.item())
                #print('Classification accuracy: ', classification(out, label))
                #print('Time: ', end-start)

                f.write('Training Loss: '+str(loss)+"\n")
                #f.write('Classification accuracy: '+str(classification(out, label))+"\n")
                f.flush()

                writer.add_scalar("data/training_loss", loss, i)
                batch_end = time.time()
                print(i, "batch time:", batch_end - batch_start)
            epoch_end = time.time()
            print("\n")
            print('Last Training Loss: ', loss.item())
            print('Time: ', epoch_end - epoch_start)

            if validate:
                print('Testing on validation set...')
                for sample in validation_generator:
                    sample = sample.to(device)
                    #label = label.to(device)
                    out = manifold_net_con(sample)
                    #loss = criterion(out,label)
                    print("\n \n Epoch "+str(epoch)+" classification: ", classification(out, label))
                    print(loss)
                    writer.add_scalar("data/validation_loss", loss, epoch)
                    torch.save(manifold_net_con.state_dict(), 'model')

                    f.write('Validation Loss: '+str(loss)+"\n")
                    f.write('Validation Classification accuracy: '+str(classification(out, label))+"\n")

    except KeyboardInterrupt:
        pass
