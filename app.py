"""
Streamlit demo app for the Explainable AI Health Misinformation Classifier.

Ties together:
  - BERT classifier (True/Misleading/False)
  - SHAP token-level attribution
  - Chain-of-Thought explanation (Gemini)

Run locally or in Colab with a public URL via:
    streamlit run app.py & npx localtunnel --port 8501
(Colab-specific launch instructions given separately.)
"""

import os
import streamlit as st
import torch
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google import genai

LABEL_ORDER = ['True', 'Misleading', 'False']
MODEL_PATH = 'keerthinani/health_misinfo_classifier'

COT_PROMPT_TEMPLATE = """You are a fact-checking assistant analyzing a health-related claim from social media.

Claim: "{claim}"

A classifier has labeled this claim as: {label}

Think step-by-step:
1. What is the claim actually asserting?
2. What established medical/scientific evidence is relevant here?
3. Why does the evidence support labeling this claim as "{label}"?

Then provide a final explanation in 2-3 clear sentences suitable for a general
audience, similar in style to a WHO myth-busting fact check.

Respond in this exact format:
REASONING: <your step-by-step reasoning>
EXPLANATION: <your final 2-3 sentence explanation>
"""


@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return tokenizer, model, device


def build_predict_fn(tokenizer, model, device):
    def predict_fn(texts):
        inputs = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=256, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
    return predict_fn


@st.cache_resource
def get_shap_explainer(_predict_fn, _tokenizer):
    masker = shap.maskers.Text(_tokenizer)
    return shap.Explainer(_predict_fn, masker, output_names=LABEL_ORDER)


def get_cot_explanation(claim, label, api_key, model='gemini-3.6-flash'):
    client = genai.Client(api_key=api_key)
    prompt = COT_PROMPT_TEMPLATE.format(claim=claim, label=label)
    response = client.models.generate_content(model=model, contents=prompt)
    content = response.text
    if 'EXPLANATION:' in content:
        return content.split('EXPLANATION:')[1].strip()
    return content.strip()


def main():
    st.set_page_config(page_title="Health Misinformation Detector", layout="wide")
    st.title("🩺 Explainable AI Health Misinformation Detector")
    st.caption(
        "Classifies health-related social media claims as True, Misleading, "
        "or False, with SHAP-based and Chain-of-Thought explanations."
    )

    tokenizer, model, device = load_classifier()
    predict_fn = build_predict_fn(tokenizer, model, device)
    explainer = get_shap_explainer(predict_fn, tokenizer)

    st.sidebar.header("Settings")
    gemini_key = st.sidebar.text_input(
        "Gemini API key (for Chain-of-Thought explanation)",
        type="password",
        help="Get a free key at aistudio.google.com/apikey"
    )

    example_claims = [
        "Drinking hot water and gargling salt water kills the coronavirus.",
        "Says FEMA is giving essential workers $1,000.",
        "Wearing a mask reduces the risk of spreading COVID-19 to others.",
    ]
    chosen_example = st.selectbox(
        "Try an example claim, or type your own below:",
        [""] + example_claims
    )

    if 'claim_text_box' not in st.session_state:
        st.session_state.claim_text_box = ""
    if chosen_example and chosen_example != st.session_state.get('_last_example', ""):
        st.session_state.claim_text_box = chosen_example
    st.session_state['_last_example'] = chosen_example

    claim_text = st.text_area(
        "Claim to check:",
        key='claim_text_box',
        height=100,
        placeholder="Paste a health-related claim here..."
    )

    if st.button("Analyze Claim", type="primary") and claim_text.strip():
        with st.spinner("Classifying..."):
            probs = predict_fn([claim_text])[0]
            pred_idx = int(np.argmax(probs))
            pred_label = LABEL_ORDER[pred_idx]

        col1, col2, col3 = st.columns(3)
        col1.metric("True", f"{probs[0]:.1%}")
        col2.metric("Misleading", f"{probs[1]:.1%}")
        col3.metric("False", f"{probs[2]:.1%}")

        badge_color = {"True": "green", "Misleading": "orange", "False": "red"}[pred_label]
        st.markdown(f"### Prediction: :{badge_color}[{pred_label}]")

        st.subheader("🔍 SHAP Token Attribution")
        with st.spinner("Computing SHAP values..."):
            shap_values = explainer([claim_text])
            tokens = shap_values.data[0]
            values = shap_values.values[0][:, pred_idx]
            ranked = sorted(zip(tokens, values), key=lambda x: -abs(x[1]))[:10]

        for tok, val in ranked:
            direction = "pushed toward" if val > 0 else "pushed away from"
            st.write(f"**`{tok.strip()}`** — {direction} *{pred_label}* ({val:+.4f})")

        st.subheader("💬 Chain-of-Thought Explanation")
        if not gemini_key:
            st.warning("Enter a Gemini API key in the sidebar to generate this explanation.")
        else:
            with st.spinner("Generating explanation..."):
                try:
                    explanation = get_cot_explanation(claim_text, pred_label, gemini_key)
                    st.info(explanation)
                except Exception as e:
                    st.error(f"Could not generate explanation: {e}")


if __name__ == '__main__':
    main()
