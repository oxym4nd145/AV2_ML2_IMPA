import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch
import numpy as np
from io import BytesIO

class plot:
    def __init__(self, model_dict, path):
        
        self.epoch = []
        self.train = []
        self.val = []
        self.best_val = []
        self.path = path

        for i in model_dict:

            self.epoch.append(i['epoch'])

            self.train.append(i['train'])

            self.val.append(i['val'])

            self.best_val.append(i['best_val'])
    
    def show(self, save=True):

        fig, ax = plt.subplots(1, 2, figsize = (30, 10))

        ax[0].plot(self.epoch, self.train, color = 'orange', linewidth = 2, label = 'Erro de treino')
        ax[0].grid()
        ax[0].legend()

        ax[0].set_xlabel('Época')
        ax[0].set_ylabel('Erro de treino')
        ax[0].set_title('Erro de treino por época')


        ax[1].plot(self.epoch, self.val, color = 'blue', linewidth = 2, label = 'Erro de val.')
        ax[1].plot(self.epoch, self.best_val, color = 'red', linestyle = 'dashed', linewidth = 1, label = 'Melhor erro de val.')

        ax[1].grid()
        ax[1].legend()

        ax[1].set_xlabel('Época')
        ax[1].set_ylabel('Erro de validação')
        ax[1].set_title('Erro de validação por época')

        plt.show()

        if save:
            plt.savefig(self.path)

class gif:
    def __init__(self, dataset, model, label, device):

        self.model = model
        self.ds = dataset
        self.label = label
        self.device = device
    
    def show(self, steps):

        self.model.eval()

        frames = []

        for i in range(steps):

            x, y = self.ds[i]

            if len(y.shape)==4:
                y = y[0, :, :, :]

            with torch.no_grad():
                pred = self.model(x.unsqueeze(0).to(self.device))

            pred = pred.squeeze(0).cpu().numpy()

            y = y.cpu().numpy()

            fig, ax = plt.subplots(2,2, figsize=(15, 15))

            ax[0][0].imshow(y[:,:,0])
            ax[0][0].set_title("Truth")

            ax[0][1].imshow(pred[:,:,0])
            ax[0][1].set_title("Prediction")

            ax[1][0].imshow(np.abs(y[:,:,0] - y[:, :, 1]), cmap='gnuplot')
            ax[1][0].set_title("Error")

            len = pred.shape[-1]

            ax[1][1].imshow(sum([pred[:,:,i] for i in range(1, len)]))
            ax[1][1].set_title("Soma das previsões dos outros campos")

            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()

            buf.seek(0)
            frames.append(imageio.imread(buf))

        imageio.mimsave(self.label + ".gif", frames, fps=10)






