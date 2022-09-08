
import tensorflow as tf

def res_block(x, size, strides=1, reg = None, name = ''):
    idt = tf.keras.layers.Conv2D(size, (1,1), padding='same', strides=strides, kernel_regularizer=reg, bias_regularizer=reg, name = f'{name}_conv_idt')(x)
    x = bn_relu(x, name = f'{name}_bnrelu_1')
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(size, (3,3), padding='same', strides = strides, kernel_regularizer=reg, bias_regularizer=reg, name = f'{name}_conv_1')(x)
    x = bn_relu(x, name = f'{name}_bnrelu_2')
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(size, (3,3), padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = f'{name}_conv_2')(x)
    return tf.keras.layers.Add(name = f'{name}_add')([x, idt])
    
def se_block(x, ch_size):
    sp_exc = tf.keras.layers.Conv2D(1, (1,1), padding='same',  activation='sigmoid')(x)
    sp_out = tf.keras.layers.Multiply()([sp_exc , x])

    ch_exc = tf.keras.layers.GlobalAvgPool2D()(x)
    ch_exc = tf.keras.layers.Dense(int(ch_size/2), use_bias=False)(ch_exc)
    ch_exc = tf.keras.layers.Dense(int(ch_size), activation='sigmoid', use_bias=False)(ch_exc)
    ch_out = tf.keras.layers.Multiply()([ch_exc , x])

    return tf.math.maximum(sp_out, ch_out)

def se_block_mod(x_in, x_mask, ch_size):
    sp_exc = tf.keras.layers.Conv2D(1, (1,1), padding='same',  activation='sigmoid')(x_mask)
    sp_out = tf.keras.layers.Multiply()([sp_exc , x_in])

    ch_exc = tf.keras.layers.GlobalAvgPool2D()(x_mask)
    ch_exc = tf.keras.layers.Dense(int(ch_size/2), use_bias=False)(ch_exc)
    ch_exc = tf.keras.layers.Dense(int(ch_size), activation='sigmoid', use_bias=False)(ch_exc)
    ch_out = tf.keras.layers.Multiply()([ch_exc , x_in])

    return tf.math.maximum(sp_out, ch_out)

def bn_relu(x, name = ''):
    x = tf.keras.layers.BatchNormalization(name = f'{name}_bn')(x)
    return tf.keras.layers.Activation('relu', name=f'{name}_relu')(x)

def resunet_base(input, model_size, n_output, reg_weight, name):
    x = tf.keras.layers.Conv2D(model_size[0], (3,3), padding='same', name = f'{name}_e0_conv_1')(input)
    x = bn_relu(x, name = f'{name}_e0_bnrelu')
    x = tf.keras.layers.Conv2D(model_size[0], (3,3), padding='same', name = f'{name}_e0_conv_2')(x)
    idt = tf.keras.layers.Conv2D(model_size[0], (1,1), padding='same', name = f'{name}_e0_conv_idt')(input)
    e1 = tf.keras.layers.Add(name = f'{name}_e0_add')([x, idt])
    
    e2 = res_block(e1, model_size[1], 2, name = f'{name}_e1')
    
    e3 = res_block(e2, model_size[2], 2, name = f'{name}_e2')
    
    bt = res_block(e3, model_size[3], 2, name = f'{name}_e3')
    
    d3 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_3')(bt)
    d3 = tf.keras.layers.Concatenate(name = f'{name}_concat_3')([d3, e3])
    
    d2 = res_block(d3, model_size[2], 1, name = f'{name}_d3')
    
    d2 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_2')(d2)
    d2 = tf.keras.layers.Concatenate(name = f'{name}_concat_2')([d2, e2])
    
    d1 = res_block(d2, model_size[1], 1, name = f'{name}_d2')
    
    d1 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_1')(d1)
    d1 = tf.keras.layers.Concatenate(name = f'{name}_concat_1')([d1, e1])
    
    d0 = res_block(d1, model_size[0], 1, name = f'{name}_d1')
    
    output = tf.keras.layers.Conv2D(n_output, (1,1), padding='same', name = f'{name}_conv_classify')(d0)
    return tf.keras.layers.Activation('softmax', name = f'{name}_activation_softmax')(output)

def resunet_encoder(input_layer, model_size, reg_weight, name):
    x = tf.keras.layers.Conv2D(model_size[0], (3,3), padding='same', name = f'{name}_e0_conv_1')(input_layer)
    x = bn_relu(x, name = f'{name}_e0_bnrelu')
    x = tf.keras.layers.Conv2D(model_size[0], (3,3), padding='same', name = f'{name}_e0_conv_2')(x)
    idt = tf.keras.layers.Conv2D(model_size[0], (1,1), padding='same', name = f'{name}_e0_conv_idt')(input_layer)
    e1 = tf.keras.layers.Add(name = f'{name}_e0_add')([x, idt])
    
    e2 = res_block(e1, model_size[1], 2, name = f'{name}_e1')
    
    e3 = res_block(e2, model_size[2], 2, name = f'{name}_e2')
    
    bt = res_block(e3, model_size[3], 2, name = f'{name}_e3')

    return bt, e3, e2, e1

def resunet_decoder(encoder_outs, model_size, reg_weight, name):
    bt, e3, e2, e1 = encoder_outs

    d3 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_3')(bt)
    if e3 is not None: d3 = tf.keras.layers.Concatenate(name = f'{name}_concat_3', axis=-1)([d3, e3])
    
    d2 = res_block(d3, model_size[2], 1, name = f'{name}_d3')
    
    d2 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_2')(d2)
    if e2 is not None: d2 = tf.keras.layers.Concatenate(name = f'{name}_concat_2', axis=-1)([d2, e2])
    
    d1 = res_block(d2, model_size[1], 1, name = f'{name}_d2')
    
    d1 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_1')(d1)
    if e1 is not None: d1 = tf.keras.layers.Concatenate(name = f'{name}_concat_1', axis=-1)([d1, e1])
    
    d0 = res_block(d1, model_size[0], 1, name = f'{name}_d1')

    return d0

def resunet_classifier(decoder_outs, model_size, n_output, reg_weight, name):
    fusion_output = res_block(decoder_outs, model_size[0], 1, name = f'{name}_resblock')
    output = tf.keras.layers.Conv2D(n_output, (1,1), padding='same', activation='softmax', name = f'{name}_lastconv_softmax')(fusion_output)

    return output


def resunet(shape, model_size, n_output, reg_weight, name):

    input_layer = tf.keras.Input(shape, name=f'{name}_input')
    encoder_outs = resunet_encoder(input_layer, model_size, reg_weight, name = f'{name}_encoder')
    decoder_outs = resunet_decoder(encoder_outs, model_size, reg_weight, name = f'{name}_decoder')
    classifier_out = resunet_classifier(decoder_outs, model_size, n_output, reg_weight, name = f'{name}_classifier')

    return tf.keras.models.Model(inputs=input_layer, outputs = classifier_out)
 
def early_fusion_resunet(shape_opt, shape_sar, model_size, n_output, reg_weight, name):

    #optical stream
    opt_input= tf.keras.Input(shape_opt, name=f'{name}_input') 
    opt_encoder_outs = resunet_encoder(opt_input, model_size, reg_weight, name=f'{name}_opt_encoder')

    #sar encoder
    sar_input= tf.keras.Input(shape_sar) 
    sar_encoder_outs = resunet_encoder(sar_input, model_size, reg_weight, name=f'{name}_sar_encoder')

    #fusion
    fus_out = tf.keras.layers.Concatenate(name = f'{name}_concat_fus', axis = -1)([opt_encoder_outs[0], sar_encoder_outs[0]])
    e3 = e2 = e1 = None #no skip connections
    fus_encoder_outs = (fus_out, e3, e2, e1)

    fus_decoder_outs = resunet_decoder(fus_encoder_outs, model_size, reg_weight, name = f'{name}_fus_decoder')
    fus_classifier_out = resunet_classifier(fus_decoder_outs, model_size, n_output, reg_weight, name = f'{name}_fus_classifier')

    return tf.keras.models.Model(inputs=[opt_input, sar_input], outputs = fus_classifier_out)


def late_fusion_resunet(shape_opt, shape_sar, model_size, n_output, reg_weight, name):

    #optical stream
    opt_input= tf.keras.Input(shape_opt, name=f'{name}_input') 
    opt_encoder_outs = resunet_encoder(opt_input, model_size, reg_weight, name=f'{name}_opt_encoder')
    opt_decoder_outs = resunet_decoder(opt_encoder_outs, model_size, reg_weight, name = f'{name}_opt_decoder')

    #sar encoder
    sar_input= tf.keras.Input(shape_sar)
    sar_encoder_outs = resunet_encoder(sar_input, model_size, reg_weight, name=f'{name}_sar_encoder')
    sar_decoder_outs = resunet_decoder(sar_encoder_outs, model_size, reg_weight, name = f'{name}_sar_decoder')

    #fusion
    fus_decoder_outs = tf.keras.layers.Concatenate(name = f'{name}_concat_fus', axis = -1)([opt_decoder_outs, sar_decoder_outs])
    fus_classifier_out = resunet_classifier(fus_decoder_outs, model_size, n_output, reg_weight, name = f'{name}_fus_classifier')

    return tf.keras.models.Model(inputs=[opt_input, sar_input], outputs = fus_classifier_out)

def opt_post_cmap_resunet(model_1, model_2, shape_cmap, model_size, n_output, reg_weight, name):
    input_1 = model_1.input
    input_2 = model_2.input
    input_3 = tf.keras.Input(shape_cmap, name = 'cmap_input')

    model_1.trainable = False
    model_2.trainable = False

    model_1_out = model_1.get_layer(index=-2).output
    model_2_out = model_2.get_layer(index=-2).output

    concat_fus = tf.keras.layers.Concatenate(name='fus_cmap_0')([model_1_out, model_2_out, input_3])
    concat_fus = tf.keras.layers.Conv2D(model_size[0], (1,1), padding='same', name = f'{name}_conv_fus')(concat_fus)
    concat_fus = se_block(concat_fus, model_size[0])
    cmap_encoder_outs = resunet_encoder(concat_fus, model_size, reg_weight, name=f'{name}_cmap_encoder')
    cmap_decoder_outs = resunet_decoder(cmap_encoder_outs, model_size, reg_weight, name = f'{name}_cmap_decoder')
    cmap_classifier_out = resunet_classifier(cmap_decoder_outs, model_size, n_output, reg_weight, name = f'{name}_cmap_classifier')

    return tf.keras.Model(inputs = [input_1, input_2, input_3], outputs = cmap_classifier_out)

#def ef_post_cmap_resunet(model_1, model_2, shape_cmap, model_size, n_output, reg_weight, name):
