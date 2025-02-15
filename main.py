from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain import LLMChain
import streamlit as st
import os

# Set your Google API key using Streamlit secrets
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Streamlit App Title with Emojis and Colors
st.markdown(
    """
    <style>
    .title {
        font-size: 40px !important;
        color: #FF4B4B !important;
        text-align: center;
    }
    .subheader {
        font-size: 20px !important;
        color: #1F51FF !important;
        text-align: center;
    }
    .itinerary-output {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        white-space: pre-line; /* Preserve line breaks in itinerary */
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">✈️ Travel Itinerary Generator - VK ✈️</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subheader">Plan your perfect trip with AI 🌍</p>',
    unsafe_allow_html=True
)

# Create prompt template for generating travel itineraries
itinerary_template = """
Create a detailed {days}-day travel itinerary for {destination} from {start_date} to {end_date}. 
The traveler is interested in {interests} and has a budget of {budget}. 
Include activities, places to visit, and food recommendations.
"""

itinerary_prompt = PromptTemplate(
    template=itinerary_template,
    input_variables=['days', 'destination', 'start_date', 'end_date', 'interests', 'budget']
)

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Create LLM chain using the prompt template and model
itinerary_chain = LLMChain(llm=gemini_model, prompt=itinerary_prompt)

# Streamlit UI
st.markdown("### 🎨 Customize Your Travel Itinerary")
destination = st.text_input("Enter your destination (e.g., Paris, Bali): 🌍")
start_date = st.date_input("Select your travel start date: 📅")
end_date = st.date_input("Select your travel end date: 📅")
interests = st.text_input("Enter your interests (e.g., hiking, museums, food): 🏞️")
budget = st.selectbox(
    "Select your budget range: 💰",
    ["Low", "Medium", "High"]
)

# Calculate the number of days
if start_date and end_date:
    days = (end_date - start_date).days
    st.write(f"Your trip duration: {days} days 🗓️")

if st.button("Generate Travel Itinerary ✈️", key="generate_button"):
    if destination and start_date and end_date and interests and budget:
        # Invoke the chain to generate the travel itinerary
        with st.spinner("Generating your travel itinerary... 🌍"):
            itinerary = itinerary_chain.run({
                "days": days,
                "destination": destination,
                "start_date": start_date,
                "end_date": end_date,
                "interests": interests,
                "budget": budget
            })
        
        # Display the generated travel itinerary with styling
        st.markdown("### 🗺️ Your Travel Itinerary:")
        st.markdown(f'<div class="itinerary-output">{itinerary}</div>', unsafe_allow_html=True)
    else:
        st.error("❌ Please provide all the required details.")