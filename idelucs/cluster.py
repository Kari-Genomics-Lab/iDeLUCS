import sys
import numpy as np

from .utils import SummaryFasta
from . import models as models



class iDeLUCS_cluster():
    def __init__(self, sequence_file, n_clusters=5, n_epochs=500, 
                 n_mimics=3, batch_sz=512, k=4, weight=0.25, n_voters=1):
        
        self.args = dict()
        self.args['sequence_file'] = sequence_file
        self.args['n_clusters'] = n_clusters

        self.args['n_epochs'] = n_epochs
        self.args['n_mimics'] = n_mimics
        self.args['batch_sz'] = batch_sz

        self.args['GT_file'] = None
        self.args['k'] = k

        self.args['optimizer'] = "RMSprop"
        self.args['lambda'] = 2.8
        self.args['weight'] =  weight
        self.args['n_voters'] = n_voters
        self.args["lr"] = 1e-3  #5e-4
        self.args["model_size"] = "linear"  #5e-4
        self.args['scheduler'] = None
        
    def fit_predict(self, kmers):
        model = models.IID_model(self.args)
        model.names, model.lengths, model.GT, model.cluster_dis = SummaryFasta(model.sequence_file,
                                                                               model.GT_file)
        
        model.build_dataloader()
        predictions = []

        for voter in range(self.args['n_voters']):
            sys.stdout.write(f"\r........... Training Model ({voter+1}/{ self.args['n_voters']})................")
            sys.stdout.flush()
            model.net.apply(models.weights_init)
            model.epoch = 0
            model_loss = []
            for i in range(self.args['n_epochs']):
                loss = model.contrastive_training_epoch()
                model_loss.append(loss)

            length = len(model.names)
            y_pred, probabilities, latent = model.predict()

        return y_pred, latent