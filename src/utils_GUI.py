#GUI.input_n_epochs.
#GUI.input_n_mimics.
#GUI.input_n_clusters.
#GUI.ChooseSeq_Button.
#GUI.ChooseGT_Button.


# GUI.input_model_size.
# GUI.input_optimizer.
# GUI.input_batch_sz.
# GUI.input_lambda.
# GUI.input_noise.

def define_ToolTips(GUI):
    GUI.ChooseSeq_Button.setToolTip('Single Fasta file with training sequences.')

    GUI.ChooseGT_Button.setToolTip('Tab separated file with possible labels for the training dataset. \n'
    'This labels wont be used during trainig, just for testing purposes. \n'
    '(Check documentation for file headers)')




    GUI.input_k.setToolTip('k-mer length')

    GUI.input_n_mimics.setToolTip('Number of data augmentations per sequence \n' 
                                  'that will be considered during training')


    GUI.input_n_epochs.setToolTip('Number of training epochs. An epoch is defined \n' 
                                 'as a training iteration over all the training pairs.')

    GUI.input_n_clusters.setToolTip('Expected or maximum number of clusters to find. \n'
                                    'It should be at least n_true_clusters \n'
                                    'when GT is provided')


    #GUI.input_model_size.setToolTip('This is a tooltip for the QPushButton widget')
    GUI.input_optimizer.setToolTip('Optimizatin algorithm to train the neural network.')
    GUI.input_batch_sz.setToolTip('Number of data pairs that the network will receive simultaneouly for \n'
                                  'training. A larger batch may improve convergence but it may harm the \n'
                                  'accuracy. It must be a multiple of n_mimcs.')
    GUI.input_lambda.setToolTip('Hyperparameter to control cluster balance. \n'
                                'Use lambda: 1 when unbalanced clusters are expected \n'
                                'Use lambda: 2.5 when perfectly balanced clusters are expected \n')
    GUI.input_weight.setToolTip('Addition of noise to the network parameters to prevent overfitting \n'
                               'This parameter indicates the frequency of noise introduction during training \n'
                               'use noise: 0 for no noise during training')

    GUI.input_n_voters.setToolTip('Numer of models to consider in the final clustering assignment.')

    GUI.Submit_Button.setToolTip('Parse training Information')
    

    



