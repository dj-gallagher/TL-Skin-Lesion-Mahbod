from tensorflow import keras
    
    

def create_basic_ResNet50():
    
    """
    Returns a basic ResNet50 model with FC layers replaced.
    Optimizer = SGD
    Loss = Categorical cross entropy
    """
    
    # DEFINING MODEL LAYERS
    # ---------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(128,128,3))
    
    #base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    base_model.trainable = True # Blocks 1-17 Frozen as in Mahbod et al.
    
    
    # Define output layers (Mahbod et al. used here)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=64, 
                           activation="relu")(x)
    predictions = keras.layers.Dense(units=3, activation="softmax")(x)

    # Create model using forzen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name="Basic_ResNet50") 
    
    

    # OPTIMIZERS
    # -------------------------------------
    optimizer = keras.optimizers.SGD()
        
    
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    # Apply label smoothing factor, default is 0 (no smoothing)
    loss_func = keras.losses.CategoricalCrossentropy()
        
    metrics_list = ['accuracy',
                    keras.metrics.AUC()] 
    
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
    
    return model




def create_baseline_ResNet50(random_seed):
    """
    Returns compiled baseline ResNet50 model ready for training.
    """
        
    # DEFINING MODEL LAYERS
    # -------------------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(128,128,3))
    
    base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    
    
    # Define output layers (Mahbod et al. used here)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)

    #x = keras.layers.Dropout(rate=0.05, seed=random_seed)(x)

    x = keras.layers.Dense(units=64, 
                           activation="relu", 
                           kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=1, seed=random_seed))(x)
    
    #x = keras.layers.Dropout(rate=0.05, seed=random_seed)(x)

    
    predictions = keras.layers.Dense(units=3, activation="softmax",
                                     kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=1, seed=random_seed))(x)

    # Create model using frozen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name="Baseline_ResNet50") 
    
    # UNFREEZE 17TH BLOCK
    # -------------------------------------
    # Create dictionary of layer name and whether the layer is trainable 
    trainable_dict = dict([ (layer.name, layer.trainable) for layer in model.layers ])
    
    # Identify names of layers in 17th block
    block_17_names = []
    for name in [ layer.name for layer in model.layers ]: # iterate through model layer names
        if "conv5_block3" in name: # conv5_block3 is naming schemee for 17th block
            block_17_names.append(name)
            
    # Set these layers to be trainable
    for name in block_17_names:
        trainable_dict[name] = True  # change dict values to true     
    
    for layer_name, trainable_bool in trainable_dict.items():
        layer = model.get_layer(name=layer_name)
        layer.trainable = trainable_bool

    
    # OPTIMIZER
    # -------------------------------------
    
    # Different LR for pretrained and FC layers
    pretrained_lr = 0.0001 
    FC_lr = 10 * pretrained_lr 
    
    # ----- STANDARD OPTIMIZERS ------
    optimizer = keras.optimizers.Adam(learning_rate=pretrained_lr)
    #optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    #optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
        
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    loss_func = keras.losses.CategoricalCrossentropy()
    
    metrics_list = ['accuracy',
                    keras.metrics.AUC()] 
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
    
    return model
        
        
        

def compile_improved_ResNet50(random_seed,
                              enable_dropout,
                              dropout_rate,
                              label_smoothing_factor,
                              enable_cosineLR):
    
    
    """
    Returns compiled ResNet50 model with improvements ready for training.
    
    Improvements:
    - Dropoit on FC layers 
    - Cosine learning rate decay
    - Smoothing of OneHot labels
    
    """
        
        
        
    # DEFINING MODEL LAYERS
    # -------------------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(128,128,3))
    
    base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    
    
    # Define output layers (Mahbod et al. used here)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)

    if enable_dropout:
        x = keras.layers.Dropout(rate=dropout_rate, seed=random_seed)(x)

    x = keras.layers.Dense(units=64, 
                           activation="relu", 
                           kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=1, seed=random_seed))(x)
    
    if enable_dropout:
        x = keras.layers.Dropout(rate=dropout_rate, seed=random_seed)(x)

    
    predictions = keras.layers.Dense(units=3, activation="softmax",
                                     kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=1, seed=random_seed))(x)

    # Create model using frozen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name="Improved_ResNet50") 
    
    
    
    # UNFREEZE 17TH BLOCK
    # -------------------------------------
    # Create dictionary of layer name and whether the layer is trainable 
    trainable_dict = dict([ (layer.name, layer.trainable) for layer in model.layers ])
    
    # Identify names of layers in 17th block
    block_17_names = []
    for name in [ layer.name for layer in model.layers ]: # iterate through model layer names
        if "conv5_block3" in name: # conv5_block3 is naming schemee for 17th block
            block_17_names.append(name)
            
    # Set these layers to be trainable
    for name in block_17_names:
        trainable_dict[name] = True  # change dict values to true     
    
    for layer_name, trainable_bool in trainable_dict.items():
        layer = model.get_layer(name=layer_name)
        layer.trainable = trainable_bool
    
    

    # COSINE LR DECAY
    # -------------------------------------
    # Different LR for pretrained and FC layers
    pretrained_lr = 0.0001 
    FC_lr = 10 * pretrained_lr 
    
    
    if enable_cosineLR:
        # decay steps = (batches per epoch) * (number of epochs)
        steps = 66 * (75)
        
        # Cosine learning rate decay 
        lr_decay_function = keras.experimental.CosineDecay(initial_learning_rate=0.001,
                                                            decay_steps=steps,
                                                            alpha=0.001*0.01) # minimum learning rate
        
        decayed_optimizer = keras.optimizers.SGD(learning_rate=lr_decay_function, momentum=0.9)


    
    # OPTIMIZER
    
    if enable_cosineLR:
        optimizer = decayed_optimizer
    else:
        # ----- STANDARD OPTIMIZERS ------
        optimizer = keras.optimizers.Adam(learning_rate=pretrained_lr)
        #optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
        #optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
        
        
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    # Apply label smoothing factor, default is 0 (no smoothing)
    #loss_func = keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth_factor)
    loss_func = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor)
    
    metrics_list = ['accuracy',
                    keras.metrics.AUC()] 
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
    
    return model
    
        