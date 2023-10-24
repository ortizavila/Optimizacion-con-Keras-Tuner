###########################################################
## Referencias:                                          ##
# https://www.tensorflow.org/guide/gpu                   ##
# https://www.tensorflow.org/tutorials/images/cnn        ##
# https://www.tensorflow.org/tutorials/keras/keras_tuner ##
# https://keras.io/keras_tuner/                          ##
##########****************************#####################

def model_builder(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (4, 4), activation='relu', input_shape=(756,120, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (10, 10), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (4, 4), activation='relu'))

    
    model.add(layers.Flatten())
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    #hp_units = hp.Int('units', min_value=32, max_value=512, step=32)

    #model.add(layers.Dense(units=hp_units, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4,1e-5])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='/home/user/',
                     project_name='microSeisms6')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

with tf.device('/device:GPU:2'):
    tuner.search(np.append(X_pos,X_neg,axis=0).reshape(4100+4000,756,120), np.append(X_label_1,X_label_0,axis=0),
                 epochs=50, validation_split=0.2, callbacks=[stop_early])#

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is  and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    #{best_hps.get('units')} 


# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(np.append(X_micro,X_nomicro,axis=0).reshape(4000+41000,756,120), np.append(X_label_1,X_label_0,axis=0), 
                    epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(np.append(X_pos,X_neg,axis=0).reshape(2*(40*100+780),756,120), np.append(X_label_1,X_label_0,axis=0),
               epochs=best_epoch, validation_split=0.2)


