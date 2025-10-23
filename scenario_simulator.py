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
# NOTE: we keep a guest_mood in the seed for generation guidance, but we do not display the mood in the UI.
scenarios = {
    "Late check-out request": {
        "guest_mood": "polite but direct",
        "context_seed": "Guest requests a late check-out, but hotel is at full capacity for the next day."
    },
    "Guest upset about room cleanliness": {
        "guest_mood": "polite but direct",
        "context_seed": "Guest found hair in the bathroom and the trash wasn't emptied."
    },
    "VIP upgrade negotiation": {
        "guest_mood": "polite but direct",
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


# ----------------- Utility helpers -----------------
def _model_error_allows_fallback(err_str: str, current_model: str, models_to_try: list):
    """Return True if the error string looks like a model access problem and a fallback remains."""
    err = (err_str or "").lower()
    model_issues = (
        "not found",
        "does not exist",
        "model_not_found",
        "you do not have access",
        "permission denied",
    )
    if any(tok in err for tok in model_issues):
        # allow fallback if current_model is not the last in models_to_try
        if current_model != models_to_try[-1]:
            return True
    return False


# ----------------- API interactions -----------------
def generate_scenario_context_and_message(scenario_name: str, model: str = DEFAULT_MODEL):
    """
    Use the API to generate a vivid scenario context and a matching guest initial message.
    Returns dict {scenario_context, guest_message, raw, used_model}
    """
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Make sure OPENAI_API_KEY is set.")

    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    seed = scenarios[scenario_name]["context_seed"]
    mood = scenarios[scenario_name].get("guest_mood", "polite but direct")

    prompt = f"""
You are a concise, practical curriculum writer for a hotel training simulator. Given the scenario seed and guest mood, generate:
1) A short narrative Scenario Context (3-4 sentences) that sounds real and not flowery — give concrete details (time of day, service constraints, brief backstory) so the learner can picture the situation.
2) A descriptive Guest Message (one sentence, up to two) that is polite but direct and includes an unanticipated, plausible reason that makes the guest's request feel urgent.

Return EXACTLY a JSON object with keys "scenario_context" and "guest_message" and nothing else.
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
                max_tokens=400,
            )
            raw_content = _get_text_from_response(response)
            used_model = attempt_model
            break
        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            # detect model access errors and try fallback
            if _model_error_allows_fallback(err_str, attempt_model, models_to_try):
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
1) Reply AS THE GUEST with a realistic in-character response (brief, 1-3 sentences). The guest should be polite but direct; if their request was not accommodated, include an unanticipated but plausible reason that increases urgency.
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
            if _model_error_allows_fallback(err_str, attempt_model, models_to_try):
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
You are an expert training coach for hotel associates. Given the scenario context, the guest's initial message, the guest's reply to the associate's first response, the associate's ORIGINAL response, and the associate's REVISED response, do the following:

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
            if _model_error_allows_fallback(err_str, attempt_model, models_to_try):
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

# Optionally show model info (default: hidden). Set SHOW_MODEL_INFO=1|true|yes to reveal.
SHOW_MODEL_INFO = os.getenv("SHOW_MODEL_INFO", "false").lower() in ("1", "true", "yes")

# Represent scenarios as clickable tabs (tabs show the scenario name in the tab label; inside the tab we show an Overview)
tabs = st.tabs(list(scenarios.keys()))

# Ensure session state keys
if "scenario_name" not in st.session_state:
    st.session_state["scenario_name"] = None
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# For each tab, show an Overview and a small practice note. Do NOT show the guest's mood in the UI.
for idx, (name, data) in enumerate(scenarios.items()):
    with tabs[idx]:
        st.subheader("Overview")
        st.write(data.get("context_seed"))
        st.write(
            "Why practice this: short roleplays like this help you quickly test phrasing that preserves brand tone while resolving urgent guest needs."
        )
        if st.button(f"Use this scenario: {name}", key=f"use_scenario_{idx}"):
            st.session_state["scenario_name"] = name
            # Force regeneration of scenario context/message
            try:
                gen = generate_scenario_context_and_message(name)
                st.session_state["scenario_context"] = gen["scenario_context"]
                st.session_state["guest_message"] = gen["guest_message"]
                st.session_state["scenario_raw"] = gen["raw"]
                st.session_state["scenario_model"] = gen.get("used_model")
                # initialize chat messages with system context and guest opening
                st.session_state["chat_messages"] = [
                    {"role": "system", "text": st.session_state["scenario_context"]},
                    {"role": "assistant", "text": st.session_state["guest_message"]},
                ]
                # clear previous results
                for k in ["guest_reply", "feedback", "raw_simulation", "original_response", "revised_response", "analysis_result"]:
                    if k in st.session_state:
                        del st.session_state[k]
            except Exception as e:
                st.error(f"Failed to generate scenario: {e}")

# "Scenario" header (formerly "Conversation") with Refresh button placed beside it
col_left, col_right = st.columns([10, 1])
col_left.subheader("Scenario")
# Place the Refresh Scenario button to the right of the header
if col_right.button("Refresh Scenario"):
    if st.session_state.get("scenario_name"):
        try:
            gen = generate_scenario_context_and_message(st.session_state["scenario_name"])
            st.session_state["scenario_context"] = gen["scenario_context"]
            st.session_state["guest_message"] = gen["guest_message"]
            st.session_state["scenario_raw"] = gen["raw"]
            st.session_state["scenario_model"] = gen.get("used_model")
            st.session_state["chat_messages"] = [
                {"role": "system", "text": st.session_state["scenario_context"]},
                {"role": "assistant", "text": st.session_state["guest_message"]},
            ]
            for k in ["guest_reply", "feedback", "raw_simulation", "original_response", "revised_response", "analysis_result"]:
                if k in st.session_state:
                    del st.session_state[k]
        except Exception as e:
            st.error(f"Failed to refresh scenario: {e}")
    else:
        st.warning("No scenario is active. Choose a scenario tab and click 'Use this scenario' first.")

# Chat area (conversation-like)
chat_box = st.container()
with chat_box:
    # Render messages
    for msg in st.session_state.get("chat_messages", []):
        role = msg.get("role", "assistant")
        text = msg.get("text", "")
        try:
            with st.chat_message(role):
                st.write(text)
        except Exception:
            # fallback for older streamlit
            if role == "user":
                st.write(f"You: {text}")
            elif role == "system":
                st.write(f"Context: {text}")
            else:
                st.write(text)

# User sends original response via chat_input
user_input = st.chat_input("Respond to the guest (press Enter to send)...")
if user_input:
    # append user message
    st.session_state["chat_messages"].append({"role": "user", "text": user_input})
    st.session_state["original_response"] = user_input
    # call simulation
    with st.spinner("Generating guest reply and feedback..."):
        try:
            guest_reply, feedback, raw = simulate_guest_response(
                user_input,
                st.session_state.get("scenario_context", ""),
                st.session_state.get("guest_message", ""),
            )
            st.session_state["guest_reply"] = guest_reply
            st.session_state["feedback"] = feedback
            st.session_state["raw_simulation"] = raw

            # append assistant guest reply
            st.session_state["chat_messages"].append({"role": "assistant", "text": guest_reply or "(No parsed guest reply)"})
            # append coach feedback as assistant message prefixed
            coach_text = (feedback or "(No parsed feedback)")
            st.session_state["chat_messages"].append({"role": "assistant", "text": f"Coach: {coach_text}"})
        except Exception as e:
            st.error(f"Simulation failed: {e}")

# If we have guest reply and feedback, allow revised attempt via chat_input
if st.session_state.get("guest_reply") is not None:
    revised_input = st.chat_input("Try a revised response (apply feedback) — press Enter to submit...", key="revised_input")
    if revised_input:
        # append revised user message
        st.session_state["chat_messages"].append({"role": "user", "text": revised_input})
        st.session_state["revised_response"] = revised_input
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

                # append coach analysis message
                score = analysis_result.get("score")
                analysis = analysis_result.get("analysis") or ""
                recommendations = analysis_result.get("recommendations") or ""
                coach_response = ""
                if score is not None:
                    coach_response += f"Score (0-100): {score}\n\n"
                coach_response += analysis + "\n\nRecommendations: " + recommendations
                st.session_state["chat_messages"].append({"role": "assistant", "text": coach_response})
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# Show raw generated scenario output only when debugging is explicitly enabled
if SHOW_MODEL_INFO and st.session_state.get("scenario_raw"):
    with st.expander("Show raw generated scenario output"):
        st.code(st.session_state["scenario_raw"])

# Raw simulation output (kept for debugging)
if st.session_state.get("raw_simulation"):
    with st.expander("Show raw simulation output"):
        st.code(st.session_state["raw_simulation"])

# Raw analyzer output (kept for debugging)
if st.session_state.get("analysis_result"):
    with st.expander("Show raw analyzer output"):
        st.code(st.session_state.get("analysis_result", {}).get("raw", ""))
