# libs
import tensorflow as tf
import numpy as np
import os

# for building dataset
from_tensor_slices = tf.data.Dataset.from_tensor_slices

# for builbing models
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Embedding = tf.keras.layers.Embedding
GRU = tf.keras.layers.GRU
LSTM = tf.keras.layers.LSTM

# ### ### ### ### ### ### ### ### ### ### ##
# datasets
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# ### ### ### ### ### ### ### ### ### ### ##

class Data_Set:
  def __init__(
      self, 
      file_path: str, 
      sequence_length: int = 100,
      batch_size: int = 64,
      buffer_size: int = 10000,
      one_hot: bool = True,
      verbose: bool = False
      ):

    # full data text 
    self.text = open(file_path, 'rb').read().decode(encoding='utf-8')
    # unique characters in data text
    self.chars = sorted(set(self.text))
    self.unique_char_count = len(self.chars)
    # hash table for character to integer and array for int to char
    self.char_to_int = {u:i for i, u in enumerate(self.chars)}
    self.int_to_char = np.array(self.chars)
    # for shuffling the training data
    self.batch_size = batch_size
    self.buffer_size = buffer_size
    # data set to train on
    self.sequence_length = sequence_length
    self.unshuffled_data = None
    self.training_data = None
    self.make_tensor_data()

    if(verbose):
      print(f'Length of text: {len(self.text)} characters\n{len(self.chars)} unique characters')

  # converts text to tensor trainign data
  def make_tensor_data(self):

    def split_input_target(chunk):
      input_text = chunk[:-1]
      target_text = chunk[1:]
      return input_text, target_text

    # data text converted to int as array
    text_to_int = np.array([self.char_to_int[c] for c in self.text])

    create_training_data = from_tensor_slices(text_to_int)
    training_sequences = create_training_data.batch(self.sequence_length + 1, drop_remainder=True)

    self.training_data = training_sequences.map(split_input_target)
    self.shuffle()
  
  # shuffles the training dataset
  def shuffle(self):
    self.training_data = self.training_data.shuffle(self.buffer_size)

# ### ### ### ### ### ### ### ### ### ### ##
# models
# ### ### ### ### ### ### ### ### ### ### ##

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def build_model(
    data_set,
    file_name: str = "./model_weights_saved.hdf5",
    embedding_dim: int = 256,
    rnn_units: int = 1024,
    batch_size: int = 64,
    verbose: bool = False
  ):
  model = Sequential()
  model.add(Embedding(data_set.unique_char_count, embedding_dim, batch_input_shape=[batch_size, None]))
  model.add(LSTM(rnn_units, stateful=True, recurrent_initializer='glorot_uniform', return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(rnn_units, stateful=True, recurrent_initializer='glorot_uniform', return_sequences=True))
  model.add(Dropout(0.2))
  model.add(Dense(data_set.unique_char_count))
  if os.path.exists(file_name): model.load_weights(file_name)
  model.compile(optimizer='adam', loss=loss)
  if verbose: model.summary()

  return model

# ### ### ### ### ### ### ### ### ### ### ##
# prediction
# ### ### ### ### ### ### ### ### ### ### ##

def generate_text(build_model_function, data_set, start_string, file_name: str = "model_weights_saved.hdf5", temperature=1.0, num_generate=1000):
  predict_model = build_model_function(data_set, file_name=file_name, batch_size=1)
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  # num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [data_set.char_to_int[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperature results in more predictable text.
  # Higher temperature results in more surprising text.
  # Experiment to find the best setting.
  # temperature = 1.0

  # Here batch size == 1
  predict_model.reset_states()
  for _ in range(num_generate):
    predictions = predict_model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the predict_model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # Pass the predicted character as the next input to the predict_model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(data_set.int_to_char[predicted_id])

  return (start_string + ''.join(text_generated))

ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

EPOCHS = 0

data_set = Data_Set(
  file_path='./texts/websters.txt',
  verbose=True  
)

file_name = "model_weights_saved.hdf5"

model = build_model(data_set, file_name=file_name, verbose=True)

checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')

def predict():
  print(generate_text(build_model, data_set, start_string=u'CRUSTECEAN\n'))

callbacks = [checkpoint]

history = model.fit(data_set.training_data, epochs=EPOCHS, callbacks=callbacks)

temperature = 1.0
num_generate = 1000
generated_text = ''
num_mores = 1

def execute_command(str):
  global temperature
  global num_generate
  global generated_text
  global num_mores
  if str[1:2] == 't':
    try:
      temperature = float(str[3:])
      print(f'(｡◕‿◕｡)\nchanged temp to {temperature}\n')
      input_text()
    except ValueError:
      print(f'I know a number when I see one, and \'{str[3:]}\' is definately not a number!\nPlease Try again.\n')
      input_text()
  elif str[1:2] == 'n':
    try:
      num_generate = int(str[3:])
      print(f'(｡◕‿◕｡)\nnumbers to generate changed to: {num_generate}\n')
      input_text()
    except ValueError:
      print(f'(  ⚆ _ ⚆ )\nI know a number when I see one, and \'{str[3:]}\' is definately not a number!\nPlease Try again.\n')
      input_text()
  elif str[1:2] == 'm':
    num_mores += 1
    generated_text = generate_text(build_model, data_set, start_string=generated_text, temperature=temperature, num_generate=num_generate * num_mores)
    print(generated_text)
    input_text()
  elif str[1:2] == 'q':
    return print('(｡-_-｡ )人( ｡-_-｡)\nGOODBYE FRIENDO\nHAVE A NICE DAY\n')
  else:
    print(f'ლ(ಠ益ಠლ)\ninvalid command\n')
    input_text()
  
def input_text():
  global generated_text
  global num_mores
  welcome_string = '\n\n└[∵┌]**********└[ ∵ ]┘**********[┐∵]┘\nWELCOME TO THE AI DICTIONARY\n'
  variable_string = f'my prediction tempurature is: {temperature}\nI will generate {num_generate} characters per prediction\n'
  prompt_string = 'enter a word and I will use my neural nets to give you a definition\n>'

  input_prediciton = input(welcome_string + variable_string + prompt_string)
  print('└[∵┌]**********└[ ∵ ]┘**********[┐∵]┘\n')

  if input_prediciton[:1] == '\\':
    execute_command(input_prediciton)
  else:
    num_mores = 1
    input_prediciton = input_prediciton.upper() + '\n'
    # print(input_prediciton)
    generated_text = generate_text(build_model, data_set, start_string=input_prediciton, temperature=temperature, num_generate=num_generate)
    print(generated_text)
    # print(generated_text.find('\r\n\r\n'))
    input_text()

input_text()