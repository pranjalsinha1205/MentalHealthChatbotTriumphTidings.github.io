import nltk   # Import the NLTK library for natural language processing
import numpy as np  # Import NumPy for scientific calculations
import json  # Import JSON for handling JSON files
import random  # Import random for generating random responses
from keras.models import Sequential  # Import Keras for building the neural network model
from keras.layers import Dense, Embedding, LSTM, Dropout
from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer for word lemmatization

# Load intents file
with open('intents.json', 'r') as file:  # Open and read the intents JSON file
    intents = json.load(file)  # Load the JSON data into the 'intents' variable

# Extract words, classes, and documents from intents file
words = []  # Initialize an empty list to store tokenized words
classes = []  # Initialize an empty list to store intent classes
documents = []  # Initialize an empty list to store patterns and corresponding intents
ignore_words = ['?', '!']  # Define a list of punctuation marks to ignore
lemmatizer = WordNetLemmatizer()  # Initialize WordNetLemmatizer object for word lemmatization

# Iterate through intents and patterns to tokenize words and extract classes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)  # Add tokenized words to the 'words' list
        documents.append((w, intent['tag']))  # Add patterns and corresponding intents to 'documents'
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add intent classes to 'classes' if not already present

# Lemmatize words and remove duplicates and sort the lists
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]  # Lemmatize and remove duplicates
words = sorted(list(set(words)))  # Sort the list of words

# Sort the list of classes
classes = sorted(list(set(classes)))

# Create training data
training = []  # Initialize an empty list to store training data
output_empty = [0] * len(classes)  # Create a list of zeros with length equal to the number of classes

# Create a bag of words for each pattern and corresponding intent
for doc in documents:
    bag = []  # Initialize an empty list to store the bag of words for the pattern
    pattern_words = doc[0]  # Get words from the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]  # Lemmatize words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)  # Create a binary bag of words for the pattern

    output_row = list(output_empty)  # Create a copy of the list with zeros
    output_row[classes.index(doc[1])] = 1  # Set the index corresponding to the intent class to 1
    training.append([bag, output_row])  # Add the bag of words and intent class to the training data

# Pad the sublists to make them equal in length
max_length = len(max(training, key=lambda x: len(x[0]))[0])  # Find the length of the longest bag of words
for sublist in training:
    sublist[0] += [0] * (max_length - len(sublist[0]))  # Pad the bag of words with zeros to match the length

random.shuffle(training)  # Shuffle the training data for randomness
training = np.array(training, dtype=object)  # Convert the training data to a NumPy array

# Split data into training and testing sets
train_x = list(training[:, 0])  # Get the bag of words as the input data
train_y = list(training[:, 1])  # Get the corresponding intent classes as the output data

# Build the neural network model
model = Sequential()  # Create a sequential model
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # Add a dense layer with ReLU activation
model.add(Dropout(0.5))  # Add dropout regularization to prevent overfitting
model.add(Dense(64, activation='relu'))  # Add another dense layer with ReLU activation
model.add(Dropout(0.5))  # Add dropout regularization
model.add(Dense(len(train_y[0]), activation='softmax'))  # Add a dense layer with softmax activation for output

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data for 100 epochs with batch size 5
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Function to clean and tokenize user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize words in the sentence
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lemmatize words
    return sentence_words  # Return the cleaned and tokenized words

# Function to create a bag of words from user input
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)  # Clean and tokenize user input
    bag = [0] * len(words)  # Initialize a bag of words with zeros
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1  # Set the corresponding index to 1 if word is in the bag
                if show_details:
                    print(f"Found in bag: {w}")  # Print debug information if show_details is True
    return np.array(bag)  # Return the bag of words as a NumPy array

# Function to predict the class of the user input and get a response
def predict_class(sentence):
    p = bow(sentence, words, show_details=False)  # Get the bag of words for the user input
    res = model.predict(np.array([p]))[0]  # Use the trained model to predict the intent probabilities
    ERROR_THRESHOLD = 0.25  # Define an error threshold for intent classification
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filter intents above the threshold

    results.sort(key=lambda x: x[1], reverse=True)  # Sort intents by probability in descending order
    return_list = []  # Initialize an empty list to store intent predictions
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})  # Add intent and probability to list
    return return_list  # Return the list of intent predictions

# Function to handle user input and generate a response
def get_response(user_input):
    response = "I'm sorry, I don't understand."  # Default response if intent is not recognized
    ints = predict_class(user_input)  # Get intent predictions for the user input
    tag = ints[0]['intent']  # Get the most probable intent
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])  # Get a random response from the matched intent
            break  # Exit the loop after finding the matched intent
    return response  # Return the generated response to the user input
