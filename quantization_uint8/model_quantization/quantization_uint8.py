import tensorflow as tf

saved_model_dir = "../../model_file/pb_model_resnet18"



converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir,
                                                     input_arrays=["inputs"],
                                                     input_shapes={"inputs": [1, 256,192,3]},)
                                                     # output_arrays=["predictions"])
converter.quantized_input_stats = {'inputs' : (0., 1.)}
converter.inference_type = tf.uint8
# converter.optimizations = ["DEFAULT"]  # 保存为v1,v2版本时使用
# converter.post_training_quantize = True  # 保存为v2版本时使用
converter.default_ranges_stats=[-1,1]
converter.inference_input_type=tf.uint8 # optional
converter.inference_output_type=tf.uint8
tflite_model = converter.convert()
# open("tflite_model_v3/test_graph_uint8.tflite", "wb").write(tflite_model)

# Save the model.
with open('../../model_file/tflite_model/test_resnet18_uint8.tflite', 'wb') as f:
  f.write(tflite_model)