import tensorflow as tf
from lib import Data_Set, build_model, generate_text

ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

EPOCHS = 0

data_set = Data_Set(
  file_path='./texts/lovecraft_complete.txt',
  verbose=True  
)

file_name = "./models/lovecraft_model.hdf5"

model = build_model(data_set, file_name=file_name, verbose=True)

checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')

def predict():
  print(generate_text(build_model, data_set, start_string=u'CRUSTECEAN\n'))

callbacks = [checkpoint]

history = model.fit(data_set.training_data, epochs=EPOCHS, callbacks=callbacks)

temperature = 0.8
num_generate = 500
generated_text = ' '
num_mores = 1

def execute_command(str):
  global temperature
  global num_generate
  global generated_text
  global num_mores
  if str[1:2] == 't':
    try:
      temperature = float(str[3:])
      print(f'\n(｡◕‿◕｡)\nchanged temp to {temperature}\n')
      input_text()
    except ValueError:
      print(f'\nI know a number when I see one, and \'{str[3:]}\' is definately not a number!\nPlease Try again.\n')
      input_text()
  elif str[1:2] == 'n':
    try:
      num_generate = int(str[3:])
      print(f'\n(｡◕‿◕｡)\nnumbers to generate changed to: {num_generate}\n')
      input_text()
    except ValueError:
      print(f'\n(  ⚆ _ ⚆ )\nI know a number when I see one, and \'{str[3:]}\' is definately not a number!\nPlease Try again.\n')
      input_text()
  elif str[1:2] == 'm':
    num_mores += 1
    print(f'\n(｡◕‿◕｡)\none moment please...\n')
    generated_text = generate_text(build_model, data_set, file_name=file_name, start_string=generated_text, temperature=temperature, num_generate=num_generate * num_mores)
    print(generated_text)
    input_text()
  elif str[1:2] == 'q':
    return print('\n(｡-_-｡ )人( ｡-_-｡)\nGOODBYE FRIENDO\nHAVE A NICE DAY\n')
  else:
    print(f'\nლ(ಠ益ಠლ)\ninvalid command\n')
    input_text()
  
def input_text():
  global generated_text
  global num_mores
  welcome_string = '\n└[∵┌]**********└[ ∵ ]┘**********[┐∵]┘\n\nWELCOME TO THE AI LOVECRAFTIAN GENERATOR\n'
  variable_string = f'\n(｡◕‿◕｡)\nmy prediction tempurature is: {temperature}\nI will generate {num_generate} characters per prediction\n'
  command_strings = '\ncommands:\n\\t to change temp\n\\n to change number of characters\n\\m for more text\n\\q to quit\n'
  end_string = '\n└[∵┌]**********└[ ∵ ]┘**********[┐∵]┘\n'
  prompt_string = '\nenter the title of a long lost lovecraftian horror and I will use my neural nets to find it for you\n>'
  input_prediciton = input(welcome_string + variable_string + command_strings + end_string + prompt_string)
  # print('└[∵┌]**********└[ ∵ ]┘**********[┐∵]┘\n')

  if input_prediciton[:1] == '\\':
    execute_command(input_prediciton)
  else:
    num_mores = 1
    input_prediciton = input_prediciton.upper() + '\n'
    # print(input_prediciton)
    print(f'\n(｡◕‿◕｡)\none moment please...\n')
    generated_text = generate_text(build_model, data_set, file_name=file_name, start_string=input_prediciton, temperature=temperature, num_generate=num_generate)
    print(generated_text)
    # print(generated_text.find('\r\n\r\n'))
    input_text()

input_text()