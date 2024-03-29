from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Mahbod LR Schedule. multiply LR by 0.1 at 5th and 10th epoch
    """

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        
        # 534 steps per epoch. Drop LR at 5th and 10th epocj
        if tf.equal( step, tf.constant( 534*5, dtype=tf.float32 ) ).numpy() == 1: 
        #if step == (534*5):
            return self.initial_learning_rate * 0.1
        
        elif tf.equal( step, tf.constant( 534*10, dtype=tf.float32 ) ).numpy() == 1: 
        #elif step == (534*10):  
            return self.initial_learning_rate * 0.1
        
        else:
            return self.initial_learning_rate 

        """tf.cond( pred=tf.equal( step, tf.constant( 534*5 ) ), true_fn=lambda: self.initial_learning_rate * 0.1, false_fn=lambda: self.initial_learning_rate )
        tf.cond( pred=tf.equal( step, tf.constant( 534*10 ) ), true_fn=lambda: self.initial_learning_rate * 0.1, false_fn=lambda: self.initial_learning_rate )"""
        
        #return tf.matmul(self.initial_learning_rate, tf.matmul( tf.floormod(step ), 0.1 ))
        
    
    
    


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
    x = keras.layers.Dense(units=64, 
                           activation="relu", 
                           kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=1, seed=random_seed))(x)
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
        
    
    
    '''# Identify names of layers in 16th block
    block_16_names = []
    for name in [ layer.name for layer in model.layers ]: # iterate through model layer names
        if "conv5_block2" in name: # conv5_block3 is naming schemee for 17th block
            block_16_names.append(name)
            
    # Set these layers to be trainable
    for name in block_16_names:
        trainable_dict[name] = True  # change dict values to true'''  
        
        
    
    for layer_name, trainable_bool in trainable_dict.items():
        layer = model.get_layer(name=layer_name)
        layer.trainable = trainable_bool   


        
    # OPTIMIZERS
    # -------------------------------------
    # Different LR for pretrained and FC layers
    pretrained_lr = 0.001 # ADAM - 0.0001 
    new_lr = 10 * pretrained_lr 

    

    # ----- MULTIOPTIMIZER ------

    optimizers = [keras.optimizers.SGD(learning_rate=MyLRSchedule(pretrained_lr), momentum=0.9),
                  keras.optimizers.SGD(learning_rate=MyLRSchedule(new_lr), momentum=0.9)]
    
    #optimizers = [keras.optimizers.Adam(learning_rate=MyLRSchedule(pretrained_lr)),
    #              keras.optimizers.Adam(learning_rate=MyLRSchedule(new_lr))]

    # Layer objects for pre-trained and FC layers
    block_17_layers = [ model.get_layer(name=name) for name in block_17_names ]
    new_fc_layers = model.layers[-3:]

    # (Optimizer, layer) pairs 
    block_17_optimizers_and_layers =  [(optimizers[0], block_17_layers)]  #[  (optimizers[0],layer) for layer in block_17_layers ]
    new_fc_optimizers_and_layers = [(optimizers[1], new_fc_layers)]  #[  (optimizers[1],layer) for layer in new_fc_layers ]
    optimizers_and_layers = block_17_optimizers_and_layers + new_fc_optimizers_and_layers

    # Optimizer with different learning rates across layers
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
    
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
        
        

def compile_improved_ResNet50(random_seed=365,
                              steps_per_epoch=None,
                              enable_dropout=False,
                              dropout_rate=0,
                              label_smoothing_factor=0,
                              enable_cosineLR=False,
                              alpha=0.1,
                              steps_multiplier=1):
    
    
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
        x = keras.layers.Dropout(rate=dropout_rate, seed=123)(x)

    x = keras.layers.Dense(units=64, 
                           activation="relu", 
                           kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=1, seed=random_seed))(x)
    
    if enable_dropout:
        x = keras.layers.Dropout(rate=dropout_rate, seed=123)(x)

    
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
    pretrained_lr = 0.001 
    FC_lr = 10 * pretrained_lr
    
    # Need two optimizers, one for FC and one for pretrained layers
    if enable_cosineLR:
        # decay steps = (steps per epoch) * (number of epochs)
        steps = steps_per_epoch * 15 
        
        steps = steps * steps_multiplier
        
        # Cosine learning rate decay fro pretrained layers
        pretrained_lr_decay_function = keras.experimental.CosineDecay(initial_learning_rate=pretrained_lr,
                                                            decay_steps=steps,
                                                            alpha=pretrained_lr*0.01) # minimum learning rate
        
        pretrained_optimizer = keras.optimizers.SGD(learning_rate=pretrained_lr_decay_function, momentum=0.9)
        
        # Cosine learning rate decay for FC layers
        FC_lr_decay_function = keras.experimental.CosineDecay(initial_learning_rate=FC_lr,
                                                            decay_steps=steps,
                                                            alpha=FC_lr*0.01) # minimum learning rate
        
        FC_optimizer = keras.optimizers.SGD(learning_rate=FC_lr_decay_function, momentum=0.9)


    
    # OPTIMIZER
    # -------------------------------------
    
    
    # ----- MULTIOPTIMIZER ------
    if enable_cosineLR:
        optimizers = [pretrained_optimizer,
                  FC_optimizer]
    else:
        optimizers = [keras.optimizers.Adam(learning_rate=pretrained_lr),
                    keras.optimizers.Adam(learning_rate=FC_lr)]

    # Layer objects for pre-trained and FC layers
    block_17_layers = [ model.get_layer(name=name) for name in block_17_names ]
    new_fc_layers = model.layers[-3:]

    # (Optimizer, layer) pairs 
    block_17_optimizers_and_layers =  [(optimizers[0], block_17_layers)]  #[  (optimizers[0],layer) for layer in block_17_layers ]
    new_fc_optimizers_and_layers = [(optimizers[1], new_fc_layers)]  #[  (optimizers[1],layer) for layer in new_fc_layers ]
    optimizers_and_layers = block_17_optimizers_and_layers + new_fc_optimizers_and_layers

    # Optimizer with different learning rates across layers
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        
        
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    # Apply label smoothing factor, default is 0 (no smoothing)
    loss_func = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor)
    
    metrics_list = ['accuracy',
                    keras.metrics.AUC()] 
    
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
    
    
    
    return model

