from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa

#### ======= ####

def get_layers(layer): 
    try: 
        return layer.layers
    except AttributeError: 
        return []

def get_mult(layer):
    #if not mult then assume 1
    try:
        return layer.lr_mult
    except AttributeError:
        return 1.
    
def assign_mult(layer, lr_mult):
    #if has mult, don't override
    try:
        layer.lr_mult 
    except AttributeError: 
        layer.lr_mult = lr_mult 
    
def get_lowest_layers(model):
    layers = get_layers(model)
    
    mult = get_mult(model)
    
    if len(layers) > 0: 
        for layer in layers: 
            #propage mult to lower layers
            assign_mult(layer, mult)
            for sublayer in get_lowest_layers(layer):
                yield sublayer
    else:
        yield model
    
def apply_mult_to_var(layer): 
    mult = get_mult(layer)
    for var in layer.trainable_variables:
        var.lr_mult = tf.convert_to_tensor(mult, tf.float32)

    return layer

def inject(model): 
    
    for layer in get_lowest_layers(model): 
        apply_mult_to_var(layer) 
    
    #get opt, move the original apply fn to a safe place, assign new apply fn 
    opt = model.optimizer
    opt._apply_gradients = opt.apply_gradients
    opt.apply_gradients = apply_gradients.__get__(opt)
    opt.testing_flag = True 
    
    return model
    
def apply_gradients(self, grads_and_vars, *args, **kwargs): 
    
    if self.testing_flag: 
        print('Training with layerwise learning rates')
        self.testing_flag = False
        
    grads = [] 
    var_list = [] 
    
    #scale each grad based on var's lr_mult
    for grad, var in grads_and_vars:
        grad = tf.math.scalar_mul(var.lr_mult, grad)
        grads.append(grad)
        var_list.append(var)
    
    grads_and_vars = list(zip(grads, var_list))

#### ======= ####


def apply_desc_lr(model):
    
    # Set multipliers for FC layers
    for layer in model.layers[-4:]:
        layer.lr_mult = 10

    inject(model)
    
    return model
    

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
    

    # COSINE LR DECAY
    # -------------------------------------
    '''
    # decay steps = (batches per epoch) * (number of epochs)
    steps = 66 * (75)
    
    # Cosine learning rate decay 
    lr_decay_function = keras.experimental.CosineDecay(initial_learning_rate=0.001,
                                                        decay_steps=steps,
                                                        alpha=0.001*0.01) # minimum learning rate
    
    optimizer = keras.optimizers.SGD(learning_rate=lr_decay_function, momentum=0.9)
    '''
        
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

    
    # OPTIMIZER
    # -------------------------------------
    # Different LR for pretrained and FC layers
    pretrained_lr = 0.0001 
    new_lr = 10 * pretrained_lr 

    # ----- MULTIOPTIMIZER ------
    optimizers = [keras.optimizers.Adam(learning_rate=pretrained_lr),
                  keras.optimizers.Adam(learning_rate=new_lr)]

    # Layer objects for pre-trained and FC layers
    block_17_layers = [ model.get_layer(name=name) for name in block_17_names ]
    new_fc_layers = model.layers[-3:]

    # (Optimizer, layer) pairs 
    block_17_optimizers_and_layers =  [(optimizers[0], block_17_layers)]  #[  (optimizers[0],layer) for layer in block_17_layers ]
    new_fc_optimizers_and_layers = [(optimizers[1], new_fc_layers)]  #[  (optimizers[1],layer) for layer in new_fc_layers ]
    optimizers_and_layers = block_17_optimizers_and_layers + new_fc_optimizers_and_layers

    # Optimizer with different learning rates across layers
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
    
    # ----- STANDARD OPTIMIZERS ------
    #optimizer = keras.optimizers.Adam(learning_rate=lr)
    #optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    #optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    
    # ---------------------------
    
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    # Apply label smoothing factor, default is 0 (no smoothing)
    #loss_func = keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth_factor)
    loss_func = keras.losses.CategoricalCrossentropy(label_smoothing=0)
    
    metrics_list = ['accuracy',
                    keras.metrics.AUC()] 
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
    
    return model
        
        
        