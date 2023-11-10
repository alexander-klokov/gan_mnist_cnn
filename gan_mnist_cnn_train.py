import torch
import matplotlib.pyplot as plt

from mnist_dataset import MnistDataset

mnist_dataset_train = MnistDataset('mnist_data/mnist_train.csv')


from gan_mnist_cnn_generator import Generator
from gan_mnist_cnn_discriminator import Discriminator

from gan_mnist_cnn_utils import generate_random_seed, label_true, label_false

if torch.cuda.is_available():
  print("using cuda:", torch.cuda.get_device_name(0))
  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = Discriminator()
D.to(device)

G = Generator()
G.to(device)

# training
epochs = 3

for j in range(epochs):
    print('training epoch', j, 'of', epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset_train:    
        D.train(image_data_tensor.to(device).view(1, 1, 28, 28), label_true)
        D.train(G.forward(generate_random_seed(100).to(device)).detach(), label_false)
        G.train(D, generate_random_seed(100).to(device), label_true)
        pass
    pass

D.save_model()
G.save_model()

# get some CUDA stats
print(torch.cuda.memory_summary(device, abbreviated=True))

# display the progress
D.plot_progress()
G.plot_progress()

plt.show()
