import tensorflow as tf
from lib import Data_Set, build_model, generate_text

ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

EPOCHS = 1

data_set = Data_Set('./texts/websters.txt')

file_name = "model_weights_saved.hdf5"
model = build_model(data_set, file_name=file_name, verbose=True)

checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks = [checkpoint]

history = model.fit(data_set.training_data, epochs=EPOCHS, callbacks=callbacks)

print(generate_text(build_model, data_set, start_string=u'CRUSTECEAN'))