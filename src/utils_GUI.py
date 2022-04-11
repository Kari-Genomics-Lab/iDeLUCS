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

    GUI.ChooseGT_Button.setToolTip('Tab separated file with possible labels \n for the training' \
    'this labels wont be used for during trainig, when \n computing' \
    'the  results sequences.')

    GUI.input_k.setToolTip('k-mer length')

    GUI.input_n_mimics.setToolTip('Number of data augmentations per sequence \n' \
                                  'that will be considered during training')

    GUI.input_n_epochs.setToolTip('Number of training epochs. An epoch is defined \n' \
                                 'as a training iteration over all the training pairs.')

    GUI.input_n_clusters.setToolTip('Expected or maximum number of clusters to find.')


    #GUI.input_model_size.setToolTip('This is a tooltip for the QPushButton widget')
    GUI.input_optimizer.setToolTip('This is a tooltip for the QPushButton widget')
    GUI.input_batch_sz.setToolTip('This is a tooltip for the QPushButton widget')
    GUI.input_lambda.setToolTip('Hyperparameter to control cluster balance')
    GUI.input_noise.setToolTip('This is a tooltip for the QPushButton widget')

    

    



