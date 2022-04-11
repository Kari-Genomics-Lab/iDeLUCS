from src import models
import time

#def threaded_task(duration):
#    for i in range(duration):
#        print("Working... {}/{}".format(i + 1, duration))
#        time.sleep(1)

def train(sequence_file, n_clusters, epochs, n_mimics,
                batch_sz, k=6, GT_file=None):

    print("Loading Model")

    model = models.IID_model(sequence_file,
                              GT_file, n_clusters,
                              n_mimics, batch_sz, k)
    print("Building dataloaders")
    model.build_dataloaders()

    for i in range(epochs):
        model.unsupervised_training_epoch()
        #self.update_state(state='PROGRESS',
        #                   meta={'current':i, 'total':epochs,
        #                         'status':"Training"})

    return model