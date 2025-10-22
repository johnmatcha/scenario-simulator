import streamlit as st
import random

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

# Simulate AI guest response and feedback
def simulate_guest_response(user_input, scenario):
    responses = {
        "Late check-out request": [
            "I understand, but I really need to rest before my flight. Is there anything you can do?",
            "That's disappointing. I was hoping for a bit more flexibility."
        ],
        "Guest upset about room cleanliness": [
            "I expected better from Wyndham. Can you send someone to clean it now?",
            "This is unacceptable. I want to speak to a manager."
        ],
        "VIP upgrade negotiation": [
            "I stay here often. I think I deserve better treatment.",
            "Other hotels always upgrade me. Why not here?"
        ]
    }
    feedback = [
        "Good job acknowledging the guest's concern.",
        "Consider expressing empathy before offering a solution.",
        "Try to align your tone with Wyndham's brand voice.",
        "You could offer alternatives to show flexibility."
    ]
    return random.choice(responses[scenario]), random.choice(feedback)

# Streamlit app layout
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
