# Running this page trains the chatbot and saves it as model.h5
# importing necessary libraries
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle                                   # To access data and convert it into bytes
import numpy as np
from keras.models import Sequential             # The model created will be sequential(Single Input/Output in a  sequence)
from keras.layers import Dense, Activation, Dropout                 # To define the number of layers, neurons and Data to be left
from keras.optimizers import SGD                # Using SDG to gain better outputs
import random


lemmatizer = WordNetLemmatizer()


words = []                                       # holds a list of tokens                   # documents = combination between patterns and intents
classes = []                                     # holds a list of tags from data.json      # classes = intents
documents = []                                   # holds a tuple of (tokens,tag)            # words = all words, vocabulary
ignore_words = ['?', '!','.',"'",'/',':']        # Ignores special characters
data_file = open('data.json').read()
json_file = json.loads(data_file)                # loads json file in json_file

for intents in json_file['intents']:
    for pattern in intents['patterns']:
        # tokenize each word
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        # add documents in the corpus
        documents.append((tokens, intents['tag']))

        # add to our classes list
        if intents['tag'] not in classes:
            classes.append(intents['tag'])

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(each_word.lower()) for each_word in words if each_word not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))


pickle.dump(words, open('texts.pkl', 'wb'))          # writes tokens in texts.pkl
pickle.dump(classes, open('labels.pkl', 'wb'))       # writes tags in labels.pkl

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word to obtain lemma
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1 for each word found and 0 for not found
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test list.
train_patterns = list(training[:, 0])
train_intents = list(training[:, 1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_patterns[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_intents[0]), activation='softmax'))

# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# defining the no of training cycles and saving the model
hist = model.fit(np.array(train_patterns), np.array(train_intents), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("model created")