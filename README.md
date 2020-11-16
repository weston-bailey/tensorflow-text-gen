# tensorflow-text-gen
text generation with tensorflow and RNN LSTM GRU neural nets

lib.py is one big file to make it easy to copy into a collab notebook

### project ideas: 
- themed lorem ipsum rest api that generates the text on request.

- twitter bot that imitates the twitter persona of a well known personality.

- dictionary that makes a definition for a supplied word.

### todo list

- [] create model prediciton loop that checks predicted words against known words from the training text, and rejects unknown words (for character based generation)
- [] compare LSTM and GRU layers for speed and accuaracy 
- [] create data pipeline for tekenizing full words, treating punctuation as individual words
  - [] clean text of punctaion, new lines and trailing spaces better
- [] create pipeline for sentance by sentance fitting
  - [] find longest sentance and pad the rest
  - [] compensate for extra trailing spaces with the prediciton loop