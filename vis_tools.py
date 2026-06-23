import matplotlib.pyplot as plt

import numpy as np


class plot:
    def __init__(self, model_dict):
        
        self.epoch = []
        self.train = []
        self.val = []
        self.best_val = []

        for i in model_dict:

            self.epoch.append(i['epoch'])

            self.train.append(i['train'])

            self.val.append(i['val'])

            self.best_val.append(i['best_val'])
    
    def show(self):

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


