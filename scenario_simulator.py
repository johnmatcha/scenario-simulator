import os
import json
import re
import streamlit as st
from openai import OpenAI

# Page config
st.set_page_config(page_title="Wyndham AI Scenario Simulator", layout="centered")

# Read API key and model from environment (allow Streamlit secrets via os.environ)
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-3.5-turbo")

# Initialize OpenAI client (only if key is provided)
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Predefined scenario categories (seed templates)
scenarios = {
    "Late check-out request": {
        "guest_mood": "polite but firm",
        "context_seed": "Guest requests a late check-out, but hotel is at full capacity for the next day."
    },
    "Guest upset about room cleanliness": {
        "guest_mood": "frustrated",
        "context_seed": "Guest found hair in the bathroom and the trash wasn't emptied."
    },
    "VIP upgrade negotiation": {
        "guest_mood": "assertive",
        "context_seed": "A loyalty member is asking for a complimentary upgrade."
    }
}


# ----------------- Parsing helpers -----------------
def _parse_json_fallback(content: str):
    """
    Attempt to parse JSON. If fails, return dict parsed from labelled fields via regex.
    """
    raw = content or ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, raw
    except Exception:
        pass

    # Try labelled regex parsing for common keys
    result = {}
    # guest reply
    m = re.search(r"guest[_\s]*reply\s*[:\-]\s*(.*?)(?:\n\s*(feedback|analysis|score)\s*[:\-]|$)", raw, re.I | re.S)
    if m:
        result["guest_reply"] = (m.group(1) or "").strip()
    # feedback / coach
    m2 = re.search(r"(feedback|analysis|coach feedback)\s*[:\-]\s*(.*?)(?:\n|$)", raw, re.I | re.S)
    if m2:
        result["feedback"] = (m2.group(2) or "").strip()
    # score
    m3 = re.search(r"score\s*[:\-]\s*(\d{1,3})", raw, re.I)
    if m3:
        try:
            result["score"] = int(m3.group(1))
        except Exception:
            pass

    return result, raw


def _get_text_from_response(response):
    """
    Extract text content defensively from different client response shapes.
    """
    content = None
    try:
        content = response.choices[0].message.content
    except Exception:
        try:
            content = response.choices[0].text
        except Exception:
            content = str(response)
    return content


# ----------------- API interactions -----------------
def generate_scenario_context_and_message(scenario_name: str, model: str = DEFAULT_MODEL):
    """
    Use the API to generate a vivid scenario context and a matching guest initial message.
    Returns dict {scenario_context, guest_message, raw}
    """
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Make sure OPENAI_API_KEY is set.")

    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    seed = scenarios[scenario_name]["context_seed"]
    mood = scenarios[scenario_name].get("guest_mood", "neutral")

    prompt = f"""
You are a creative curriculum writer for a hotel training simulator. Given the following scenario seed and guest mood, generate:
1) A richly detailed, instantly relatable Scenario Context (2-4 sentences). Make it engaging, give small concrete details (time of day, service constraints, guest backstory) so the learner can picture the interaction.
2) A short Guest Message (one or two sentences) that naturally flows from the scenario context and conveys the guest's tone.

Return EXACTLY a JSON object with keys "scenario_context" and "guest_message". Do not include any extra explanation or text.
Seed: {seed}
Guest mood: {mood}
Scenario name: {scenario_name}
Example output:
{{"scenario_context": "A concise vivid context...", "guest_message": "Hi, I..."}}
"""

    models_to_try = [model]
    if FALLBACK_MODEL and FALLBACK_MODEL != model:
        models_to_try.append(FALLBACK_MODEL)

    last_exception = None
    raw_content = None
    used_model = None
    for attempt_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=attempt_model,
                messages=[
                    {"role": "system", "content": "You generate structured JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=300,
            )
            raw_content = _get_text_from_response(response)
            used_model = attempt_model
            break
        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            # detect model access errors and try fallback
            if ("model" in err_str and ("not found" in err_str or "does not exist" in err_str or "model_not_found" in err_str or "you do not have access" in err_str)) and attempt_model != models_to_try[-1]:
                continue
            else:
                raise RuntimeError(f"OpenAI API request failed: {e}")

    if used_model is None:
        raise RuntimeError(f"OpenAI API request failed after trying models {models_to_try}: {last_exception}")

    # parse
    parsed, raw = _parse_json_fallback(raw_content)
    # normalize keys
    scenario_context = parsed.get("scenario_context") or parsed.get("context") or ""
    guest_message = parsed.get("guest_message") or parsed.get("guest") or parsed.get("guest_message_text") or ""
    return {"scenario_context": scenario_context.strip(), "guest_message": guest_message.strip(), "raw": raw, "used_model": used_model}


def simulate_guest_response(user_input: str, scenario_context: str, guest_message: str, model: str = DEFAULT_MODEL):
    """
    Sends the prompt to OpenAI and returns (guest_reply, feedback, raw_model_output).
    Guest reply will show slightly amplified emotions (irritation if not accommodated, jubilation if accommodated).
    """
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Make sure OPENAI_API_KEY is set.")

    prompt = f"""
You are simulating a hotel guest and providing concise coaching feedback for a Wyndham associate.

Scenario Context:
{scenario_context}

Guest initial message:
{guest_message}

Associate response:
{user_input}

Task:
1) Reply AS THE GUEST with a realistic in-character response (brief, 1-3 sentences). If the associate's response does not accommodate the guest's request or feels dismissive, the guest should show slightly amplified irritation. If the associate offers a clear, satisfactory solution or compensation, the guest should show jubilation/relief. Be realistic.
2) Provide short coaching feedback for the associate covering empathy, tone, clarity, and brand alignment.

RETURN FORMAT:
Return a single JSON object and nothing else, for example:
{{"guest_reply": "The guest reply text", "feedback": "Coaching feedback here"}}
If you cannot return valid JSON, still include labelled lines:
Guest Reply: ...
Feedback: ...
"""

    models_to_try = [model]
    if FALLBACK_MODEL and FALLBACK_MODEL != model:
        models_to_try.append(FALLBACK_MODEL)

    last_exception = None
    raw_content = None
    used_model = None
    for attempt_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=attempt_model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that replies as a hotel guest and provides coaching feedback. Output must be a single JSON object if possible."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=600,
            )
            raw_content = _get_text_from_response(response)
            used_model = attempt_model
            break
        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            if ("model" in err_str and ("not found" in err_str or "does not exist" in err_str or "model_not_found" in err_str or "you do not have access" in err_str)) and attempt_model != models_to_try[-1]:
                continue
            else:
                raise RuntimeError(f"OpenAI API request failed: {e}")

    if used_model is None:
        raise RuntimeError(f"OpenAI API request failed after trying models {models_to_try}: {last_exception}")

    parsed, raw = _parse_json_fallback(raw_content)
    guest_reply = (parsed.get("guest_reply") or parsed.get("Guest Reply") or parsed.get("guest") or "").strip()
    feedback = (parsed.get("feedback") or parsed.get("Feedback") or parsed.get("coaching_feedback") or "").strip()

    # Annotate if fallback model used
    if used_model != model:
        raw = f"(NOTE: original model '{model}' not available; used fallback '{used_model}')\n\n{raw or ''}"

    return guest_reply, feedback, raw


def analyze_user_responses(original_response: str, revised_response: str, scenario_context: str, guest_message: str, guest_reply: str, model: str = DEFAULT_MODEL):
    """
    Ask the model to score the user's revised response and provide analysis.
    Returns dict with keys score (int 0-100), analysis (string), recommendations (string), raw.
    """
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Make sure OPENAI_API_KEY is set.")

    prompt = f"""
You are an expert training coach for hotel associates. Given the scenario context, the guest's initial message, the guest's reply to the associate's first response, the associate's ORIGINAL response, and the associate's REVISED response, provide:

1) A numeric score from 0 to 100 evaluating the REVISED response on empathy, tone, clarity, problem-solving, and brand alignment (weight equally).
2) A short analysis explaining why the score was given (2-4 sentences).
3) Concise actionable recommendations the associate can apply immediately.

Return EXACTLY a JSON object with keys:
{{"score": <int 0-100>, "analysis": "<text>", "recommendations": "<text>"}}

Scenario Context:
{scenario_context}

Guest initial message:
{guest_message}

Guest reply to the associate's ORIGINAL response:
{guest_reply}

Associate ORIGINAL response:
{original_response}

Associate REVISED response:
{revised_response}
"""

    models_to_try = [model]
    if FALLBACK_MODEL and FALLBACK_MODEL != model:
        models_to_try.append(FALLBACK_MODEL)

    last_exception = None
    raw_content = None
    used_model = None
    for attempt_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=attempt_model,
                messages=[
                    {"role": "system", "content": "You are a coach that returns a single JSON object scoring and explaining the response."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            raw_content = _get_text_from_response(response)
            used_model = attempt_model
            break
        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            if ("model" in err_str and ("not found" in err_str or "does not exist" in err_str or "model_not_found" in err_str or "you do not have access" in err_str)) and attempt_model != models_to_try[-1]:
                continue
            else:
                raise RuntimeError(f"OpenAI API request failed: {e}")

    if used_model is None:
        raise RuntimeError(f"OpenAI API request failed after trying models {models_to_try}: {last_exception}")

    parsed, raw = _parse_json_fallback(raw_content)
    # normalize and coerce
    score = parsed.get("score")
    try:
        if score is not None:
            score = int(score)
    except Exception:
        score = None

    analysis = parsed.get("analysis") or parsed.get("feedback") or ""
    recommendations = parsed.get("recommendations") or parsed.get("recommendation") or ""

    # annotate if fallback model used
    if used_model != model:
        raw = f"(NOTE: original model '{model}' not available; used fallback '{used_model}')\n\n{raw or ''}"

    return {"score": score, "analysis": analysis.strip(), "recommendations": recommendations.strip(), "raw": raw}


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

# Offer a refresh button to re-roll scenario context and guest message
refresh_button = st.button("Refresh Scenario")

# Regenerate scenario context/guest message when selection changes or refresh pressed
if "scenario_name" not in st.session_state:
    st.session_state["scenario_name"] = None

need_regen = refresh_button or (st.session_state.get("scenario_name") != selected_scenario) or ("scenario_context" not in st.session_state)
if need_regen:
    try:
        gen = generate_scenario_context_and_message(selected_scenario)
        st.session_state["scenario_context"] = gen["scenario_context"]
        st.session_state["guest_message"] = gen["guest_message"]
        st.session_state["scenario_raw"] = gen["raw"]
        st.session_state["scenario_name"] = selected_scenario
        st.session_state["scenario_model"] = gen.get("used_model")
        # Clear previous simulation results
        for k in ["guest_reply", "feedback", "raw_simulation", "original_response", "revised_response", "analysis_result"]:
            if k in st.session_state:
                del st.session_state[k]
    except Exception as e:
        st.error(f"Failed to generate scenario: {e}")
        st.stop()

# Display context and guest message
st.subheader("Scenario Context")
st.write(st.session_state.get("scenario_context", ""))

st.subheader("Guest Message")
st.write(st.session_state.get("guest_message", ""))

st.info(f"Primary model: {DEFAULT_MODEL}  •  Fallback model: {FALLBACK_MODEL}")

# User response text area and button to trigger simulation
user_response = st.text_area("Your Response (what you'd say to the guest):", height=150, key="original_response")
simulate = st.button("Simulate")

if simulate:
    if not st.session_state.get("original_response") or not st.session_state.get("original_response").strip():
        st.warning("Please enter a response before simulating.")
    else:
        with st.spinner("Generating guest reply and feedback..."):
            try:
                guest_reply, feedback, raw = simulate_guest_response(
                    st.session_state["original_response"].strip(),
                    st.session_state["scenario_context"],
                    st.session_state["guest_message"],
                )
                st.session_state["guest_reply"] = guest_reply
                st.session_state["feedback"] = feedback
                st.session_state["raw_simulation"] = raw

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

            except Exception as e:
                st.error(f"Simulation failed: {e}")

# If we have guest reply and feedback, allow the user to submit a revised attempt
if st.session_state.get("guest_reply") is not None:
    st.subheader("Try Again — Revised Response")
    revised_response = st.text_area("Your Revised Response (apply feedback, be concise):", height=150, key="revised_response")
    submit_revised = st.button("Submit Revised Response")

    if submit_revised:
        if not st.session_state.get("revised_response") or not st.session_state.get("revised_response").strip():
            st.warning("Please enter a revised response before submitting.")
        else:
            with st.spinner("Scoring and analyzing your revised response..."):
                try:
                    analysis_result = analyze_user_responses(
                        st.session_state.get("original_response", ""),
                        st.session_state.get("revised_response", ""),
                        st.session_state.get("scenario_context", ""),
                        st.session_state.get("guest_message", ""),
                        st.session_state.get("guest_reply", ""),
                    )
                    st.session_state["analysis_result"] = analysis_result

                    st.subheader("Overall Score")
                    score = analysis_result.get("score")
                    if score is not None:
                        st.metric("Score (0-100)", score)
                    else:
                        st.write("_No numeric score parsed. See full analysis below._")

                    st.subheader("Analysis")
                    st.write(analysis_result.get("analysis") or "_No analysis parsed._")

                    st.subheader("Recommendations")
                    st.write(analysis_result.get("recommendations") or "_No recommendations parsed._")

                    with st.expander("Show raw analyzer output"):
                        st.code(analysis_result.get("raw", ""))
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

# Expanders to show raw outputs for debugging
if st.session_state.get("scenario_raw"):
    with st.expander("Show raw generated scenario output"):
        st.code(st.session_state["scenario_raw"])

if st.session_state.get("raw_simulation"):
    with st.expander("Show raw simulation output"):
        st.code(st.session_state["raw_simulation"])
