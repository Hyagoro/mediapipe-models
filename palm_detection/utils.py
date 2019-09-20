import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

def set_pretrained_weights(model, weights_path='pretrained_weights/palm_detection_weights.npy'):
    # Get layer names
    layer_names = [
    'classificator_8', 'classificator_16', 'classificator_32', 
    'regressor_8', 'regressor_16', 'regressor_32',
    'conv2d', 'depthwise_conv2d', 'conv2d_transpose', 'conv2d_transpose_1'
    ]
    num_conv = 41
    num_depth_conv = 40
    # Append conv2d layer names
    for i in range(1, num_conv+1):
        name = 'conv2d_' + str(i)
        layer_names.append(name)
    for i in range(1, num_depth_conv+1):
        name = 'depthwise_conv2d_' + str(i)
        layer_names.append(name)
    layer_names.sort()

    # Set pretrained weights from npy file
    weights_dict = np.load(weights_path).item()
    for name in layer_names:
        pretrained_weights = []
        kernel_weight = weights_dict.get(name + '_Kernel')
        bias_weight = weights_dict.get(name + '_Bias')
        if name.find("conv2d_transpose") == -1:
            kernel_weight = kernel_weight.transpose(1, 2, 3, 0)
        else:
            kernel_weight = kernel_weight.transpose(1, 2, 0, 3)
        
        pretrained_weights.append(kernel_weight)
        pretrained_weights.append(bias_weight)
        layer = model.get_layer(name)
        layer.set_weights(pretrained_weights)
    
    print("[INFO] Set all pretrained weights")

def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        for idx, n in enumerate(node.input):
            print(u'└─── %d ─ %s' % (idx, n))

def convert_to_pb(model, out_path):
    sess = K.get_session()
    output_names = [node.op.name for node in model.outputs]
    frozen_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)
    with tf.gfile.GFile(out_path, 'w') as f:
        f.write(frozen_def.SerializeToString())
    print("[INFO] Save frozen graph model in %s"%(out_path))

def convert_to_tflite(pb_file_path, out_path):
    input_names = ['input_1']
    output_names = ['regressors/concat', 'classificators/concat']
    converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_file_path, input_names, output_names)
    tflite_model = converter.convert()
    open(out_path, "wb").write(tflite_model)
    print("[INFO] Save TFLite model in %s"%(out_path))

