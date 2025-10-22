import streamlit as st
import os
from openai import OpenAI

# Initialize OpenAI client using secret key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define predefined scenarios
scenarios = {
    "Late check-out request": {
        "guest_mood": "polite but firm",
        "context": "Guest requests a late check-out, but hotel is at full capacity for the next day.",
        "initial_guest_message": "Hi, I was wondering if I could get a late check-out tomorrow?"
    },
    "Guest upset about room cleanliness": {
        "guest_mood": "frustrated",
        "context": "Guest found hair in the bathroom and the trash wasn't emptied.",
        "initial_guest_message": "I'm really disappointed. My room wasn't cleaned properly!"
    },
    "VIP upgrade negotiation": {
        "guest_mood": "assertive",
        "context": "A loyalty member is asking for a complimentary upgrade.",
        "initial_guest_message": "I'm a Diamond member. Can I get a suite upgrade tonight?"
    }
}

# Function to simulate AI guest response and feedback using GPT
def simulate_guest_response(user_input, scenario):
    prompt = f"""
    You are simulating a hotel guest for Wyndham training.
    Scenario: {scenario}
    Guest mood and context: {scenarios[scenario]['guest_mood']} - {scenarios[scenario]['context']}
    Guest initial message: {scenarios[scenario]['initial_guest_message']}
    Associate response: {user_input}

    Task:
    1. Reply as the guest in a realistic tone.
    2. Provide coaching feedback for the associate (empathy, tone, brand alignment).
    Format:
    Guest Reply: <reply>
    Feedback: <feedback>
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI guest and coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    output = response.choices[0].message.content
    guest_reply = output.split("Feedback:")[0].replace("Guest Reply:", "").strip()
    feedback = output.split("Feedback:")[1].strip()

    return guest_reply, feedback

# Streamlit UI
st.title("Wyndham AI Scenario Simulator")

# Scenario selection
selected_scenario = st.selectbox("Choose a guest scenario:", list(scenarios.keys()))

# Display scenario context and initial guest message
st.subheader("Scenario Context")
st.write(scenarios[selected_scenario]["context"])

st.subheader("Guest Message")
st.write(f"Guest ({scenarios[selected_scenario]['guest_mood']}): {scenarios[selected_scenario]['initial_guest_message']}")

# User input
user_response = st.text_input("Your Response:")

# Simulate AI response and feedback
if user_response:
    ai_reply, feedback = simulate_guest_response(user_response, selected_scenario)
    st.subheader("AI Guest Reply")
    st.write(f"Guest: {ai_reply}")
    st.subheader("AI Coach Feedback")
    st.write(feedback)
