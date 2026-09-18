"""
Streamlit demo app for the Explainable AI Health Misinformation Classifier.

Ties together:
  - BERT classifier (True/Misleading/False)
  - SHAP token-level attribution, aggregated to whole words
  - Chain-of-Thought explanation (Gemini), reached INDEPENDENTLY of the
    classifier, with explicit reconciliation when the two disagree

CHANGES FROM v1 (all three are discussed in the dissertation):

1. UNCONDITIONED EXPLAINER. v1's prompt said "A classifier has labeled this
   claim as {label} ... why does the evidence support that label?" The
   explainer was therefore given the answer and asked to justify it: it could
   not disagree, and when the classifier was wrong it confidently defended a
   wrong verdict. v2 withholds the label so the explainer reaches its own
   conclusion.

2. DISAGREEMENT-AWARE OUTPUT. Because the two components are now independent
   they can conflict. Rather than asserting a verdict anyway, the interface
   surfaces the conflict and withholds confidence. For a health-misinformation
   tool, declining to assert under conflicting evidence is the correct
   behaviour.

3. WORD-LEVEL ATTRIBUTION. v1 displayed raw subword tokens ('trum', 'FE',
   'MA'), which are not human-readable. v2 sums subword attributions into
   whole words before display.
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

# Note: no {label} placeholder. The explainer is not told what the classifier
# decided, so its verdict is independent evidence rather than a rationalisation.
COT_PROMPT_TEMPLATE = """You are a fact-checking assistant analysing a health-related claim from social media.

Claim: "{claim}"

Reason step by step:
1. What is the claim actually asserting?
2. What established medical or scientific evidence is relevant?
3. What does that evidence imply about the claim's accuracy?

Then give your own verdict and a 2-3 sentence explanation for a general
audience, in the style of a WHO myth-buster.

Respond in exactly this format:
REASONING: <your step-by-step reasoning>
VERDICT: <True|Misleading|False>
EXPLANATION: <your 2-3 sentence explanation>
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


def aggregate_to_words(tokens, values):
    """Sum subword attributions into whole-word attributions.

    WordPiece splits 'vaccination' into 'vacc' + '##ination', and displaying
    those fragments separately makes the explanation hard to read and
    overstates the importance of arbitrary substrings. Summing the pieces
    gives one interpretable score per word.
    """
    words, scores, cur_w, cur_s = [], [], '', 0.0
    for tok, val in zip(tokens, values):
        t = str(tok)
        if t.strip() == '':
            if cur_w:
                words.append(cur_w); scores.append(cur_s); cur_w, cur_s = '', 0.0
            continue
        cur_w += (t[2:] if t.startswith('##') else t).strip()
        cur_s += float(val)
        # SHAP's Text masker marks a word boundary with a trailing space
        # ('FE', 'MA ' -> 'FEMA'); raw WordPiece marks continuation with '##'.
        # Handling both keeps this correct whichever masker is in use.
        if t != t.rstrip():
            words.append(cur_w); scores.append(cur_s); cur_w, cur_s = '', 0.0
    if cur_w:
        words.append(cur_w); scores.append(cur_s)
    return words, scores


def get_cot_explanation(claim, api_key, model='gemini-3.6-flash'):
    """Return (verdict, explanation). Verdict is None if unparseable."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model, contents=COT_PROMPT_TEMPLATE.format(claim=claim))
    content = response.text or ''

    verdict = None
    for label in LABEL_ORDER:
        if f'VERDICT: {label}'.lower() in content.lower():
            verdict = label
            break

    if 'EXPLANATION:' in content:
        explanation = content.split('EXPLANATION:')[1].strip()
    else:
        explanation = content.strip()
    return verdict, explanation


def main():
    st.set_page_config(page_title="Health Misinformation Detector", layout="wide")
    st.title("🩺 Explainable AI Health Misinformation Detector")
    st.caption(
        "Classifies health-related social media claims as True, Misleading, "
        "or False, with SHAP-based and Chain-of-Thought explanations."
    )

    with st.expander("How this works, and what it cannot do"):
        st.markdown(
            """
**Two independent stages.**

1. A **BERT classifier**, fine-tuned on 11,601 labelled claims, predicts a
   veracity class. It has no access to evidence and performs no fact-checking:
   it recognises the *linguistic patterns* of claims that fact-checkers have
   previously rated false. SHAP shows which words drove that prediction.
2. A **chain-of-thought explainer** reasons about the claim's factual accuracy
   using a large language model's own knowledge, and reaches its **own**
   verdict without being told what the classifier decided.

**Why that matters.** The two stages can disagree, and when they do this tool
reports the conflict instead of asserting a verdict. The explainer's factual
claims come from the language model's training data and are not retrieved from
a live source, so they can be wrong or out of date.

**Not a substitute for professional medical advice or a qualified fact-checker.**
            """
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

    # Examples drawn from the held-out test set, with their gold labels, so
    # the demo shows genuine model behaviour on unseen data rather than
    # hand-picked inputs. Covers all three classes.
    EXAMPLES = {
        "— select an example —": "",
        "[True] WHO 'mild cases' statistic":
            "Says 80% of novel coronavirus cases are “mild.”",
        "[True] New Zealand testing capacity":
            "New Zealand is also in a good position with its supplies for "
            "testing - current stock across the country is sufficient to "
            "enable 253190 tests.",
        "[True] New Zealand case-free period":
            "Correction: It is now 21 days since the last case in New Zealand.",
        "[Misleading] Tea prevents infection":
            "Tea has ingredients to ward off any coronavirus infection.",
        "[Misleading] Dettol foreknowledge":
            "Sanitizer manufacturer Dettol knew about the new coronavirus "
            "before it was reported.",
        "[Misleading] Flu shot ingredients":
            "A widely-circulated image claims to reveal the ingredients "
            "contained in this year's flu shots. The alleged ingredients "
            "include mercury, antifreeze, phenol, animal blood, animal "
            "viruses, and formaldehyde.",
        "[False] FEMA payment claim":
            "Says FEMA is giving essential workers $1,000.",
        "[False] Italy photograph":
            "A photo has been shared in multiple posts on Facebook and Twitter "
            "alongside a claim it shows the bodies of people who died in Italy "
            "after they became infected with the novel coronavirus, COVID-19.",
        "[Low confidence] Nigeria travel restrictions":
            "Govt. of Nigeria is restricting entry into the country for "
            "travellers from: China Italy Iran South Korea Spain Japan France "
            "Germany United States of America Norway United Kingdom "
            "Netherlands & Switzerland These are countries with > 1000 cases "
            "domestically",
    }

    chosen = st.selectbox(
        "Try an example claim (label in brackets is the gold label), "
        "or type your own below:", list(EXAMPLES.keys()))

    claim_text = st.text_area(
        "Claim to check:", value=EXAMPLES[chosen], height=100,
        placeholder="Paste a health-related claim here...")

    if st.button("Analyze Claim", type="primary") and claim_text.strip():
        with st.spinner("Classifying..."):
            probs = predict_fn([claim_text])[0]
            pred_idx = int(np.argmax(probs))
            pred_label = LABEL_ORDER[pred_idx]
            confidence = float(probs[pred_idx])

        col1, col2, col3 = st.columns(3)
        col1.metric("True", f"{probs[0]:.1%}")
        col2.metric("Misleading", f"{probs[1]:.1%}")
        col3.metric("False", f"{probs[2]:.1%}")

        # --- Chain-of-Thought first, so the two verdicts can be reconciled ----
        cot_verdict, explanation = None, None
        if gemini_key:
            with st.spinner("Generating independent explanation..."):
                try:
                    cot_verdict, explanation = get_cot_explanation(
                        claim_text, gemini_key)
                except Exception as e:
                    st.error(f"Could not generate explanation: {e}")

        # ------------------------- RECONCILIATION ----------------------------
        if cot_verdict and cot_verdict != pred_label:
            st.error(
                f"### ⚠️ Conflicting evidence — no verdict issued\n\n"
                f"The **classifier** rates this claim **{pred_label}** "
                f"({confidence:.0%} confidence), but **independent "
                f"evidence-based reasoning** concludes **{cot_verdict}**.\n\n"
                f"The two stages of this system disagree, so no single verdict "
                f"is reported. Read the explanation below and consult a "
                f"qualified fact-checker before drawing a conclusion."
            )
        elif cot_verdict:
            st.success(
                f"### Verdict: {pred_label}\n"
                f"Classifier ({confidence:.0%} confidence) and independent "
                f"reasoning agree."
            )
        else:
            badge = {"True": "green", "Misleading": "orange",
                     "False": "red"}[pred_label]
            st.markdown(f"### Classifier prediction: :{badge}[{pred_label}]")
            if not gemini_key:
                st.warning(
                    "Enter a Gemini API key in the sidebar to obtain an "
                    "independent second opinion. Without it, only the "
                    "classifier's pattern-based prediction is shown."
                )

        # --------------------------- EXPLANATIONS ----------------------------
        if explanation:
            st.subheader("💬 Chain-of-Thought Explanation")
            st.info(explanation)
            st.caption(
                "Generated independently of the classifier's prediction. "
                "Factual content comes from the language model's training data "
                "and is not retrieved from a live source."
            )

        st.subheader("🔍 Which words drove the classifier's prediction")
        with st.spinner("Computing SHAP values..."):
            shap_values = explainer([claim_text])
            words, scores = aggregate_to_words(
                shap_values.data[0], shap_values.values[0][:, pred_idx])
            ranked = sorted(zip(words, scores), key=lambda x: -abs(x[1]))[:10]

        for word, val in ranked:
            direction = "pushed toward" if val > 0 else "pushed away from"
            st.write(f"**`{word}`** — {direction} *{pred_label}* ({val:+.4f})")

        st.caption(
            "These are the words the classifier weighted most heavily. They "
            "reflect learned linguistic patterns, not verified facts."
        )


if __name__ == '__main__':
    main()
