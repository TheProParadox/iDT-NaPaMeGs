import tensorflow as tf

loaded_model = tf.keras.models.load_model("model_weights/forward_final.h5", compile = False)

print("Done Loading model")