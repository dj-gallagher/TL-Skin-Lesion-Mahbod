from tensorflow import keras
#from src.new_optimizer import *

def create_basic_ResNet50():
    
    """
    Returns the baseline ResNet50 model
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
    
    # ---------------------------
    
    
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
    # ---------------------------
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

    # Create model using forzen base layers and new FC layers
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
    

    # OPTIMIZERS
    # -------------------------------------
    '''learning_rate_multipliers = {}
    
    for layer in model.layers[:-2]:
        learning_rate_multipliers[layer.name] = 1
        print("Layer 1X: ", layer.name)
    for layer in model.layers[-2:]:
        learning_rate_multipliers[layer.name] = 10
        print("Layer 10X: ", layer.name)
        
    adam_with_lr_multipliers = Adam_lr_mult(lr=0.0001, multipliers=learning_rate_multipliers)'''
    
    
    # Standard Optimizer
    #optimizer = keras.optimizers.Adam(learning_rate=lr)
    optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    #optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    
    # ---------------------------
    
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    # Apply label smoothing factor, default is 0 (no smoothing)
    #loss_func = keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth_factor)
    loss_func = keras.losses.CategoricalCrossentropy()
    
    metrics_list = ['accuracy',
                    keras.metrics.AUC()] 
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
    
    return model
        
        
        