def learning_curve(X_train, y_train, X_valid, y_valid, step_size):
    # ********************** read hyper-parameters ************************
    hyperPar_dict = {"epochs": 1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "batch_norm": True,
    "keep_probability": 1}
    # ********************** ********************* ************************

    training_accuracy_list      = []
    training_loss_list          = []
    validation_accuracy_list    = []    
    validation_loss_list        = []
    
    for m_train in range(step_size, X_train.shape[0]+1, step_size):            
        m_valid = int(np.minimum(np.round(m_train*.15), X_valid.shape[0]))
        print("Training using {training_samples} training samples and {valid_samples} validation samples"
              .format(training_samples = m_train, valid_samples = m_valid))
        print()

        X_train, y_train = shuffle(X_train, y_train)
        X_valid, y_valid = shuffle(X_valid, y_valid)        
        #*******************************************
        # Train the model for m samples
        # ******************************************
        with tf.Session() as sess:               
            sess.run(tf.global_variables_initializer())
            results_dict_mSamples = train_model(X_train[0:m_train], y_train[0:m_train],
            X_valid[0:m_valid], y_valid[0:m_valid], 
            hyperPar_dict)            
            
            training_accuracy_list.append(results_dict_mSamples['training_accuracy'])
            training_loss_list.append(results_dict_mSamples['training_loss'])

            validation_accuracy_list.append(results_dict_mSamples['validation_accuracy'])
            validation_loss_list.append(results_dict_mSamples['validation_loss'])

    learning_curve_results = {
        "training_accuracy": np.array(training_accuracy_list),
        "training_loss": np.array(training_loss_list),
        "validation_accuracy": np.array(validation_accuracy_list),
        "validation_loss": np.array(validation_loss_list)
    }

    return learning_curve_results



    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_valid)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", N_CLASSES)


    # Mapping ClassID to traffic sign names
    signs = []
    signnames = csv.reader(open('signnames.csv', 'r'), delimiter=',')
    next(signnames,None)
    for row in signnames:
        signs.append(row[1])



    y_pos = np.arange(len(np.unique(y_train)))
    fig, ax = plt.subplots()
    ax.bar(y_pos, nhist_y_train, 0.35, align='center', alpha=0.5, color = 'b', label = 'train')
    #ax.bar(y_pos, nhist_y_valid, 0.8, align='center', alpha=0.5, color = 'g', label = 'valid')
    ax.set_ylabel('count of images in a class')
    ax.set_xlabel('class ID')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.show() 

    fig, ax = plt.subplots()
    #ax.bar(y_pos, nhist_y_train, 0.35, align='center', alpha=0.5, color = 'b', label = 'train')
    ax.bar(y_pos, nhist_y_valid, 0.8, align='center', alpha=0.5, color = 'g', label = 'valid')
    ax.set_ylabel('count of images in a class')
    ax.set_xlabel('class ID')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.show() 

    print("Distribution of different classes in training is \n", (hist_y_train))     



print(sign_names.get(sign_names.ClassId == 4))