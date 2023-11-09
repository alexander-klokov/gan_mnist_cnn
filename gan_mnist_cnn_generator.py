import torch
import torch.nn as nn
import pandas

from gan_mnist_cnn_utils import View

PATH_TO_MODEL = "model/gan_mnist_cnn_generator.pth"

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(100, 1*4*4),
            nn.LeakyReLU(0.2),
            # reshape to 4d
            View((1, 1, 4, 4)),

            nn.ConvTranspose2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=3),
            nn.Sigmoid(),
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []

        pass

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)
        d_output = D.forward(g_output)

        loss = D.loss_function(d_output, targets)

        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(title ='Loss: generator', ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass

    def save_model(self):
        torch.save(self.model.state_dict(), PATH_TO_MODEL)

    def load_model(self):
        self.model.load_state_dict(torch.load(PATH_TO_MODEL))