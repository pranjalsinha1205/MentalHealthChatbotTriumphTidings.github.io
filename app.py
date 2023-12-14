from flask import Flask, render_template, request
# render_template: Function to render HTML templates and pass data to them.
# request: Object to handle incoming request data, such as form data.
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Import SentimentIntensityAnalyzer from NLTK's vader module
import matplotlib.pyplot as plt  # Import matplotlib for creating plots

app = Flask(__name__)

# Import your chatbot code
import GUIChatbot

# Global list to store chat history including user input, bot response, and user sentiment
chat_history = []

# Create a Sentiment Intensity Analyzer object
sid = SentimentIntensityAnalyzer()

# Create a function to generate responses from your chatbot and to create the user's polarity graph
def generate_response(query):
    response = GUIChatbot.get_response(query)  # Get response from the chatbot logic
    # Analyze sentiment of the user's input using Sentiment Intensity Analyzer
    user_sentiment = sid.polarity_scores(query)['compound'] 
    
    # Append user sentiment, user input, and bot response to the chat history list
    chat_history.append({'user': query, 'bot': response, 'sentiment': user_sentiment})
    
    # Extract message numbers and user polarities for plotting
    message_numbers = list(range(1, len(chat_history) + 1))  # Generate message numbers for x-axis of the plot
    user_polarities = [entry['sentiment'] for entry in chat_history]  # Extract user polarities for y-axis
    
    # Plot user's polarity against message number using matplotlib
    plt.figure(figsize=(8, 6))  # Set the figure size for the plot
    plt.plot(message_numbers, user_polarities, marker='o', color='b', label='User Polarity')  # Plot user polarity
    plt.xlabel('Number of Observations(N)')  # Set x-axis label
    plt.ylabel('User Polarity')  # Set y-axis label
    plt.title('User Polarity Distribution')  # Set the plot title
    plt.legend()  # Display legend in the plot
    plt.savefig('static/sentiment_plot.png')  # Save the plot as an image file in the 'static' folder
    
    return response  # Return the chatbot's response to be displayed in the web application

# Define the index route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']  # Get user input from the form
        response = generate_response(query)  # Generate response using the chatbot logic and sentiment analysis
        return render_template('index.html', response=response, chat_history=chat_history)
    else:
        return render_template('index.html', chat_history=chat_history)  # Render the index.html template with chat history

# Define the about route
@app.route('/about')
def about():
    return render_template('about.html')  # Render the about.html template

# Start the Flask app if this script is run directly
if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode for development purposes
