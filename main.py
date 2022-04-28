import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# CONSTANTS
OPTIMIZE_NUMBER = 5
SDG = 0
MINI_BATCH = 1
SDG_MOMEMTUM = 2
ADA_GRAD = 3
RMS_PROP = 4
OPTIMIZER_STR = ["Stochastic Gradient Descent", "Mini Batch", "Mini Batch momentum", "Adaptive Gradient Descent", "RMSProp"]
OPTIMIZER_COLORS = ["#D50000", "#2962FF", "#FFD600", "#00C853", "#3E2723"]
# OPTIONSSSSSSSSSSSSSSSSSSSSS
optimizer_options = SDG;

# NETWORK PARAMETERS
n_epochs = 10
random_seed = 1
BATCH_SIZE_OPTIMIZER = 64
MNIST_TRAIN = False;
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Train and tests
train_losses = []
train_counter = []
test_losses = []
train_all = [[] for i in range(OPTIMIZE_NUMBER)]
train_all_counter = [[] for i in range(OPTIMIZE_NUMBER)]

class Net(nn.Module):
      
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
      self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
      self.conv2_drop = nn.Dropout2d()
      self.fc1 = nn.Linear(320, 50)
      self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
      x = x.view(-1, 320)
      x = F.relu(self.fc1(x))
      x = F.dropout(x, training=self.training)
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)


def loadMNIST(batch_size):
  return torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=MNIST_TRAIN, download=True,
                          transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                              (0.1307,), (0.3081,))
                          ])),
    batch_size=batch_size, shuffle=True)

def train(epoch, network, optimizer, tl):
  train_loader = tl
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    data, target = data.to(device), target.to(device)
    output = network(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    log_interval = 10 * (n_epochs/BATCH_SIZE_OPTIMIZER) if optimizer_options != 0 else 5000
    if batch_idx % log_interval == 0:
      # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      #   epoch, batch_idx * len(data), len(train_loader.dataset),
      #   100. * batch_idx / len(train_loader), loss.item()))
      batch_idx_size = BATCH_SIZE_OPTIMIZER if optimizer_options != 0 else 1
      temp_loss = loss.item() if loss.item() < 3 else 3
      train_losses.append(temp_loss)
      train_counter.append(
        (batch_idx*batch_idx_size) + ((epoch-1)*len(train_loader.dataset)))
      # train_all[optimizer_options].append(temp_loss)
      # train_all_counter[optimizer_options].append((batch_idx*batch_idx_size) + ((epoch-1)*len(train_loader.dataset)))

      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test(network, tl, nb_epoch=0):
  test_loader = tl
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = network(data)
      test_loss += F.cross_entropy(output, target, reduction="sum").item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  train_all[optimizer_options].append(test_loss)

  print(str(nb_epoch) + ' Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def plotLoss(test_counter):
    print(len(train_counter))
    plt.plot(train_counter, train_losses, color='#2962FF')
    plt.scatter(test_counter, test_losses, color='#D50000')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Nombre d\'exemple d\'entrainement')
    plt.ylabel('Cross entropy loss')
    plt.title(OPTIMIZER_STR[optimizer_options] + "  batch = " + str(batch_size_train))
    plt.show()

def plotAll():
    for i in range(0, OPTIMIZE_NUMBER):
      print(OPTIMIZER_STR[i])
      print(train_all[i])
      plt.plot([i for i in range(n_epochs+1)], train_all[i], color=OPTIMIZER_COLORS[i])
    plt.legend(OPTIMIZER_STR, loc='upper right')
    plt.xlabel('Nombre d\'exemple d\'entrainement')
    plt.ylabel('Cross entropy loss')
    plt.title("Comparaisons de différents algorithmes d'optimisation\n epoch="+str(n_epochs) + " batch_size="+str(BATCH_SIZE_OPTIMIZER))
    plt.show()
      

def main():
  train_loader = loadMNIST(batch_size_train)
  test_loader = loadMNIST(batch_size_test)
  train_losses = []
  train_counter = []
  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
  
  network = Net()
  network.to(device)

  # Stochastic Gradient Descent
  if optimizer_options == SDG:
    optimizer = optim.SGD(network.parameters(), lr=0.01)
  # Mini-Batch
  elif optimizer_options == MINI_BATCH:
    optimizer = optim.SGD(network.parameters(), lr=0.1)
  # Stochastic Gradient Descent avec mometum
  elif optimizer_options == SDG_MOMEMTUM:
    optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.5)
  # Adaptive Gradient Descent
  elif optimizer_options == ADA_GRAD:
    optimizer = optim.Adagrad(network.parameters())
  # RMSProp
  else :# optimizer_options == RMS_PROP:
    optimizer = optim.RMSprop(network.parameters(), lr=0.001)
  
  test(network, test_loader)
  for epoch in range(1, n_epochs + 1):
    train(epoch, network, optimizer, train_loader)
    test(network, test_loader, epoch)
  # plotLoss(test_counter)


if __name__ == '__main__':
  timers = []
  for i in range(0, OPTIMIZE_NUMBER):
    t0 = time.time()
    optimizer_options = i
    # Stochastic Gradient Descent
    if optimizer_options == SDG:
      batch_size_train = 1
      batch_size_test = 1000
    # Mini-Batch
    elif optimizer_options == MINI_BATCH:
      batch_size_train = BATCH_SIZE_OPTIMIZER
      batch_size_test = 1000
    # Stochastic Gradient Descent avec mometum
    elif optimizer_options == SDG_MOMEMTUM:
      batch_size_train = BATCH_SIZE_OPTIMIZER
      batch_size_test = 1000
    # Adaptive Gradient Descent
    elif optimizer_options == ADA_GRAD:
      batch_size_train = BATCH_SIZE_OPTIMIZER
      batch_size_test = 1000
    # RMSProp
    else :# optimizer_options == RMS_PROP:
      batch_size_train = BATCH_SIZE_OPTIMIZER
      batch_size_test = 1000
    print("\n"+OPTIMIZER_STR[i])
    main()
    timers.append(time.time() - t0)
  print(timers)
  plotAll()  