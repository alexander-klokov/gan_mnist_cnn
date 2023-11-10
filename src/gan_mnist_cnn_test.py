import matplotlib.pyplot as plt

from gan_mnist_cnn_generator import Generator
from gan_mnist_cnn_utils import generate_random_seed

G = Generator()
G.load_model()

_, axarr = plt.subplots(2,3, figsize=(16,8))
for i in range(2):
    for j in range(3):
        seed = generate_random_seed(100)
        output = G.forward(seed)
        img = output.detach().numpy().reshape(28,28)
        axarr[i,j].imshow(img, interpolation='none', cmap='gray')
        pass
    pass

plt.show()