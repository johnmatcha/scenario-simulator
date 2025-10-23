import os
import json
import re
import streamlit as st
from openai import OpenAI

# Page config
st.set_page_config(page_title="Wyndham AI Scenario Simulator", layout="centered")

# Read API key and model from environment (allow Streamlit secrets via os.environ)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# Initialize OpenAI client (only if key is provided)
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Predefined scenarios
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


def _parse_model_output(content: str):
    """
    Parse the model output. Prefer JSON, fall back to labelled text extraction,
    and finally return the raw string as feedback if nothing else works.
    Returns tuple (guest_reply, feedback, raw_text)
    """
    raw = content or ""
    guest_reply = ""
    feedback = ""

    # Try JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            guest_reply = parsed.get("guest_reply") or parsed.get("Guest Reply") or parsed.get("guestReply") or ""
            feedback = parsed.get("feedback") or parsed.get("Feedback") or ""
            return guest_reply.strip(), feedback.strip(), raw
    except Exception:
        pass

    # Fallback: labelled parsing using regex (case-insensitive)
    m = re.search(r"guest reply\s*[:\-]\s*(.*?)(?:\n\s*feedback\s*[:\-]\s*(.*)|$)", raw, re.I | re.S)
    if m:
        guest_reply = (m.group(1) or "").strip()
        feedback = (m.group(2) or "").strip()
        return guest_reply, feedback, raw

    # Another fallback: split by "Feedback:" label
    if "Feedback:" in raw:
        parts = raw.split("Feedback:", 1)
        guest_reply = parts[0].replace("Guest Reply:", "").strip()
        feedback = parts[1].strip()
        return guest_reply, feedback, raw

    # As a last resort, return empty guest_reply and raw as feedback so user can see what's returned
    return "", raw.strip(), raw


def simulate_guest_response(user_input: str, scenario_name: str, model: str = DEFAULT_MODEL):
    """
    Sends the prompt to OpenAI and returns (guest_reply, feedback, raw_model_output).
    Raises RuntimeError on configuration / API errors.
    """
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Make sure OPENAI_API_KEY is set in the environment or Streamlit secrets.")

    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = scenarios[scenario_name]

    # Strong instruction to return a JSON object to make parsing robust
    prompt = f"""
You are simulating a hotel guest and providing concise coaching feedback for a Wyndham associate.

Scenario: {scenario_name}
Guest mood: {scenario['guest_mood']}
Context: {scenario['context']}
Guest initial message: {scenario['initial_guest_message']}

Associate response:
{user_input}

Task:
1) Reply AS THE GUEST with a realistic in-character response (brief).
2) Provide short coaching feedback for the associate covering empathy, tone, and brand alignment.

RETURN FORMAT:
Return a single JSON object and nothing else, for example:
{{"guest_reply": "The guest reply text", "feedback": "Coaching feedback here"}}
If you cannot return valid JSON, still include labelled lines:
Guest Reply: ...
Feedback: ...
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that replies as a hotel guest and provides coaching feedback. Output must be a single JSON object with keys 'guest_reply' and 'feedback' if possible."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=600,
        )
    except Exception as e:
        # Surface a readable error to the caller
        raise RuntimeError(f"OpenAI API request failed: {e}")

    # Extract content defensively (different client versions may store text differently)
    content = None
    try:
        # preferred: choices[0].message.content
        content = response.choices[0].message.content
    except Exception:
        try:
            content = response.choices[0].text
        except Exception:
            # last resort: stringify entire response object
            content = str(response)

    guest_reply, feedback, raw = _parse_model_output(content)
    return guest_reply, feedback, raw


# ----------------- Streamlit UI -----------------
st.title("Wyndham AI Scenario Simulator")

# API key check and helpful UI message
if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. To use the simulator, set the environment variable OPENAI_API_KEY "
        "or add it to Streamlit secrets. The app cannot contact OpenAI without it."
    )
    st.markdown("If you don't have an API key and want me to provide a local mock mode for testing, tell me and I can add it.")
    st.stop()

# Scenario selection
selected_scenario = st.selectbox("Choose a guest scenario:", list(scenarios.keys()))

# Display context and guest message
st.subheader("Scenario Context")
st.write(scenarios[selected_scenario]["context"])

st.subheader("Guest Message")
st.write(f"Guest ({scenarios[selected_scenario]['guest_mood']}): {scenarios[selected_scenario]['initial_guest_message']}")

# User response text area and button to trigger simulation
user_response = st.text_area("Your Response (what you'd say to the guest):", height=150)
simulate = st.button("Simulate")

if simulate:
    if not user_response or not user_response.strip():
        st.warning("Please enter a response before simulating.")
    else:
        with st.spinner("Generating guest reply and feedback..."):
            try:
                guest_reply, feedback, raw = simulate_guest_response(user_response.strip(), selected_scenario)
                st.subheader("AI Guest Reply")
                if guest_reply:
                    st.write(guest_reply)
                else:
                    st.write("_No parsed guest reply. See raw output below._")

                st.subheader("AI Coach Feedback")
                if feedback:
                    st.write(feedback)
                else:
                    st.write("_No parsed feedback. See raw output below._")

                with st.expander("Show raw model output"):
                    st.code(raw)
            except Exception as e:
                st.error(f"Simulation failed: {e}")
