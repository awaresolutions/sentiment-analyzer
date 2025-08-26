import streamlit as st
import pandas as pd
import nltk

# NLTK Download: This is necessary for the sentiment analyzer.
# The `nltk.download` function is smart enough to only download the resource
# if it's not already present, so we don't need to check first.
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# This function simulates fetching data from an API.
# In a real-world scenario, you would replace this with a 'fetch' call
# to a backend service that retrieves reviews from the Google Maps API.
def fetch_park_reviews():
    """
    Simulates fetching mock data for park reviews in downtown Chicago.
    """
    mock_data = [
        {
            'id': 'p1',
            'name': 'Millennium Park',
            'reviews': [
                {'id': 1, 'text': 'The Bean is so cool! A must-see.', 'rating': 5},
                {'id': 2, 'text': 'Very crowded, but the garden is beautiful.', 'rating': 4},
                {'id': 3, 'text': 'A bit touristy, but a nice place to walk.', 'rating': 3},
                {'id': 4, 'text': 'Great for photos and people watching!', 'rating': 5},
                {'id': 5, 'text': 'Crowded and dirty. Not a fan.', 'rating': 2},
            ],
        },
        {
            'id': 'p2',
            'name': 'Grant Park',
            'reviews': [
                {'id': 6, 'text': 'Love the wide open spaces and the fountain!', 'rating': 5},
                {'id': 7, 'text': 'Perfect for a morning run. So peaceful.', 'rating': 4},
                {'id': 8, 'text': 'The lawn was a bit patchy, but a solid park.', 'rating': 3},
                {'id': 9, 'text': 'Amazing views of the city skyline. Breathtaking!', 'rating': 5},
                {'id': 10, 'text': 'There was a lot of trash on the ground.', 'rating': 2},
            ],
        },
        {
            'id': 'p3',
            'name': 'Maggie Daley Park',
            'reviews': [
                {'id': 11, 'text': 'The playground is fantastic for kids! Super fun.', 'rating': 5},
                {'id': 12, 'text': 'Really well designed and clean. Love the mini-golf.', 'rating': 5},
                {'id': 13, 'text': 'Gets very busy on weekends. The climbing wall is neat.', 'rating': 4},
                {'id': 14, 'text': 'Beautiful park, but very loud with all the kids.', 'rating': 3},
                {'id': 15, 'text': 'Too noisy for me. Wish it was more relaxing.', 'rating': 2},
            ],
        },
    ]
    return mock_data

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text, threshold):
    """
    Analyzes the sentiment of a given text using VADER and a custom threshold.
    Returns 'Positive', 'Negative', or 'Neutral'.
    """
    # VADER returns a dictionary with 'compound' score
    score = analyzer.polarity_scores(text)['compound']
    if score >= threshold:
        return 'Positive'
    elif score <= -threshold:
        return 'Negative'
    else:
        return 'Neutral'

# Set up the Streamlit page
st.set_page_config(layout="wide", page_title="Chicago Park Sentiment Analysis")

st.title("Chicago Park Sentiment Analyzer")

# Create two columns for the layout
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("A demonstration of sentiment analysis on mock park reviews using `nltk` and interactive sliders.")
    st.markdown("""
    **What is the Sentiment Threshold?**
    The sentiment threshold is a value that helps classify a text as either positive, negative, or neutral. It's used in conjunction with a sentiment analysis model like **VADER** (Valence Aware Dictionary and sEntiment Reasoner), which is what we're using here.
    The VADER model analyzes a text and gives it a **compound score**, which is a single, normalized value that ranges from -1 (most negative) to +1 (most positive). This score represents the overall sentiment of the text.
    The sentiment threshold acts as the boundary for this score.

    - If the compound score is **above** the threshold, the review is classified as **Positive**.
    - If the compound score is **below** the negative of the threshold (e.g., -threshold), the review is classified as **Negative**.
    - If the compound score falls **between** the negative and positive threshold values (e.g., between -0.2 and 0.2), the review is considered **Neutral**.

    By adjusting the slider, you're changing how strongly positive or negative a review's compound score needs to be to get classified as "Positive" or "Negative." A higher threshold makes the classification stricter, meaning a review needs a very strong sentiment to be labeled. A lower threshold makes it more lenient.
    """)

    # Fetch the mock data
    parks = fetch_park_reviews()

    # Create a dictionary for easy lookup
    park_data_map = {park['name']: park for park in parks}

    # Use a selectbox to choose which park to display
    selected_park_name = st.selectbox(
        "Select a park to analyze:",
        options=list(park_data_map.keys()),
        key="park_selector"
    )

    # Get the data for the selected park
    selected_park = park_data_map[selected_park_name]

    st.header(selected_park['name'])
    
    # Use a unique key for each slider to prevent conflicts
    threshold = st.slider(
        f"Adjust sentiment threshold for {selected_park['name']}",
        min_value=0.0,
        max_value=1.0,
        value=0.2, # A common default threshold for VADER
        step=0.05,
        key=f"slider_{selected_park['id']}"
    )
    
    # Create a DataFrame to hold review data for this park
    reviews_df = pd.DataFrame(selected_park['reviews'])
        
    # Apply sentiment analysis based on the slider value
    reviews_df['sentiment'] = reviews_df['text'].apply(lambda x: get_sentiment(x, threshold))

    # Display the reviews in an expandable section
    with st.expander("Show Reviews"):
        for index, row in reviews_df.iterrows():
            st.markdown(f"**Rating: {row['rating']}/5** | Sentiment: {row['sentiment']}")
            st.markdown(f"*{row['text']}*")
            st.markdown("---")

with col_right:
    st.subheader("Sentiment Summary")

    # Calculate sentiment counts and format for display
    sentiment_counts = reviews_df['sentiment'].value_counts().to_dict()
    
    # Ensure all three sentiment types are present for consistent charts
    sentiment_counts.setdefault('Positive', 0)
    sentiment_counts.setdefault('Neutral', 0)
    sentiment_counts.setdefault('Negative', 0)
    
    # Create columns for the metrics
    neg_col, neut_col, pos_col = st.columns(3)
    
    with neg_col:
        st.metric(label="Negative Reviews", value=sentiment_counts['Negative'])
    
    with neut_col:
        st.metric(label="Neutral Reviews", value=sentiment_counts['Neutral'])
        
    with pos_col:
        st.metric(label="Positive Reviews", value=sentiment_counts['Positive'])

    # Create a DataFrame for the bar chart
    sentiment_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Count': [sentiment_counts['Positive'], sentiment_counts['Neutral'], sentiment_counts['Negative']]
    })
    
    # Display the bar chart
    st.bar_chart(sentiment_df, x="Sentiment", y="Count", color="#007bff", use_container_width=True)
    st.markdown("---")
