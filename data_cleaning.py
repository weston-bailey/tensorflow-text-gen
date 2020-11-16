# WIP data set manipulation
import string

file_path = './texts/sonnets.txt'

text = open(file_path, 'rb').read().decode(encoding='utf-8')

# returns a hashable dict of every unique word found in a text
# only works on sonnets.txt
def every_word(text):
  # make array of words
  result = text.split(" ")

  # remove whitespace from array of words
  result[:] = [word for word in result if len(word) > 0]

  # remove newlines 
  for i in range(len(result)): result[i] = result[i].rstrip('\n')

  # remove punction
  table = str.maketrans('', '', string.punctuation)
  stripped = [word.translate(table) for word in result]

  # lower case everthing
  stripped = [word.lower() for word in stripped]

  # put words in a hashable dict
  word_dict = {}
  inc = 0
  for i in range(len(stripped)):
    if stripped[i] not in word_dict: 
      word_dict[stripped[i]] = inc
      inc += 1

  return word_dict

print(every_word(text))


# find numbers in text
def find_numbers(text):
  i = 0
  while i < len(text):
    num = None
    if text[i].isnumeric():
      num = text[i]
      j = i + 1
      while text[j].isnumeric():
        num += text[j]
        j += 1
      # do something with your number, yay!
      print(num)
      i += len(num)
    else: 
      i += 1