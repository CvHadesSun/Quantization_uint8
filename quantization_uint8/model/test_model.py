import tensorflow as tf
import re
import shutil
import tensorflow.contrib.slim as slim
from blazepose_tf1 import BlazeposeV1,BlazePoseNet
from model_conv import TestModel
from resnet18 import Resnet18

def saveTestModel():
    sess = tf.Session()
    BlazePose = Resnet18()
    variables = slim.get_variables_to_restore()
    save_vars = [variable for variable in variables if not re.search("Adam", variable.name)]
    saver = tf.train.Saver(save_vars)
    sess.run(tf.initialize_all_variables())

    # Export checkpoint to SavedModel
    builder = tf.saved_model.builder.SavedModelBuilder("../../model_file/pb_model_resnet18")
    inputs = {"inputs": tf.saved_model.utils.build_tensor_info(BlazePose.input_image)}

    outputs = {"predictions": tf.saved_model.utils.build_tensor_info(BlazePose.predictions)}

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                  method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={"serving_default": prediction_signature},
                                         legacy_init_op=legacy_init_op,
                                         saver=saver)

    builder.save()
# shutil.rmtree('../../model_file/pb_model_test')
saveTestModel()
