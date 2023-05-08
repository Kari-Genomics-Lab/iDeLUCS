def define_ToolTips(GUI):
    GUI.ChooseSeq_Button.setToolTip('Single Fasta file with training sequences. \n'
                                    'The header of each sequence must be a unique \n'
                                    'sequence identifier and each sequence must follow the \n'
                                    'IUPAC nomenclature code for nucleic acids.')

    GUI.ChooseGT_Button.setToolTip('Tab separated file with possible labels for the training dataset. \n'
    'This labels wont be used during trainig, just for testing purposes. \n'
    '(Check documentation for info on file headers)')

    GUI.input_k.setToolTip('k-mer length')

    GUI.input_n_mimics.setToolTip('Number of data augmentations per sequence \n' 
                                  'that will be considered during training')


    GUI.input_n_epochs.setToolTip('Number of training epochs. An epoch is defined \n' 
                                 'as a training iteration over all the training pairs.')

    GUI.input_n_clusters.setToolTip('Expected or maximum number of clusters to find. \n'
                                    'It should be equal or greater than n_true_clusters \n'
                                    'when GT is provided. \n'
                                    'NOTE: Use 0 for automatically finding fine-grained \n'
                                    'clusters')

    GUI.input_scheduler.setToolTip('Learning rate schedule to train the neural network.')

    GUI.input_batch_sz.setToolTip('Number of data pairs that the network will receive \n'
                                  'simultaneouly during training. A larger batch may \n'
                                  'improve convergence but it may harm the accuracy \n')

    GUI.input_lambda.setToolTip('Hyperparameter to control cluster balance. \n'
                                'Use lambda: 1.2 when unbalanced clusters are expected \n'
                                'Use lambda: 2.8 when perfectly balanced clusters are expected \n')

    GUI.input_weight.setToolTip('Relative importance of the contrastive objective on \n'
                                'the final loss. Use a higher value when low intra- \n'
                                'cluster distance is expected and a lower value when \n'
                                'high intra-cluster variability is expected')
                            

    GUI.Silhouette.setToolTip('Higher is better - This measure compares the cluster \n'
                            ' assignment of each sequence with the assignment \n'
                            'of the closest sequence assigned to a different cluster. \n'
                            'Range: [-1,1]')

    GUI.DB_Index.setToolTip('Lower is better - Measures the average distance between \n'
                            'clusters, relative to their sizes. Range: [0,2] \n')                         

    GUI.Submit_Button.setToolTip('Parse training Information')