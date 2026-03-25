# main_part1 (Streamlit) 
import io, re
from io import BytesIO
import streamlit as st
from huggingface_hub import InferenceClient
import config
# Switch provider by changing the import line:
from groq import generate_response
# from hf import generate_response 

MATH_SYSTEM = """You are a Math Mastermind.
Solve with clear step-by-step reasoning, correct notation, and a final answer.
Verify when possible; mention an alternative method briefly if relevant."""

CHAT_CSS = """
<style>
.wrap {max-height: 520px; overflow-y: auto; padding-right: 6px;}
.card{border:1px solid #e6e6e6;background:#fff;border-radius:10px;padding:14px 16px;margin:10px 0;
box-shadow:0 1px 2px rgba(0,0,0,0.04);}
.q{font-weight:700;color:#0a6ebd;margin-bottom:8px;}
.meta{display:inline-block;background:#FF9800;color:#fff;padding:2px 8px;border-radius:12px;font-size:12px;margin-left:8px}
.a{white-space:pre-wrap;color:#333;line-height:1.5;}
</style>
"""

def export_txt(history):
    txt = "".join([f"Q{i}: {h['question']}\nA{i}: {h['answer']}\n\n" for i, h in enumerate(history, 1)])
    bio = io.BytesIO(txt.encode("utf-8")); bio.seek(0); return bio

def teaching_answer(q: str) -> str:
    return generate_response(q, temperature=0.3, max_tokens=1024)

def math_answer(q: str, level: str) -> str:
    prompt = f"{MATH_SYSTEM}\n\nDifficulty: {level}\nMath Problem: {q}"
    return generate_response(prompt, temperature=0.1, max_tokens=1024)

def run_ai_teaching_assistant():
    st.title("🤖 AI Teaching Assistant")
    st.session_state.setdefault("history_ata", [])
    c1, c2 = st.columns([1, 2])
    if c1.button("🧹 Clear", key="c_ata"): st.session_state.history_ata = []; st.rerun()
    if st.session_state.history_ata:
        c2.download_button("📄 Export", export_txt(st.session_state.history_ata),
                           "AI_Teaching_Assistant_Conversation.txt", "text/plain")
    q = st.text_input("Enter your question:", key="q_ata")
    if st.button("Ask", key="a_ata"):
        if not q.strip(): st.warning("⚠️ Enter a question.")
        else:
            with st.spinner("Thinking..."):
                st.session_state.history_ata.append({"question": q.strip(), "answer": teaching_answer(q.strip())})
            st.rerun()

    if not st.session_state.history_ata: return
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    html = '<div class="wrap">'
    for i, qa in enumerate(st.session_state.history_ata, 1):
        html += f'<div class="card"><div class="q">Q{i}: {qa["question"]}</div><div class="a">{qa["answer"]}</div></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

def run_math_mastermind():
    st.title("🧮 Math Mastermind")
    st.session_state.setdefault("history_mm", [])
    st.session_state.setdefault("k_mm", 0)
    c1, c2 = st.columns([1, 2])
    if c1.button("🧹 Clear", key="c_mm"): st.session_state.history_mm = []; st.rerun()
    if st.session_state.history_mm:
        c2.download_button("📄 Export", export_txt(st.session_state.history_mm),
                           "Math_Mastermind_Solutions.txt", "text/plain")
    with st.form("mm_form", clear_on_submit=True):
        q = st.text_area("Math problem:", height=100, key=f"mm_{st.session_state.k_mm}")
        a, b = st.columns([3, 1])
        go = a.form_submit_button("Solve", use_container_width=True)
        lvl = b.selectbox("Level", ["Basic", "Intermediate", "Advanced"], index=1)
        if go:
            if not q.strip(): st.warning("⚠️ Enter a problem.")
            else:
                with st.spinner("Solving..."):
                    ans = math_answer(q.strip(), lvl)
                st.session_state.history_mm.insert(0, {"question": q.strip(), "answer": ans, "difficulty": lvl})
                st.session_state.k_mm += 1; st.rerun()

    if not st.session_state.history_mm: return
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    html = '<div class="wrap">'
    for i, qa in enumerate(st.session_state.history_mm, 1):
        html += (f'<div class="card"><div class="q">Q{i}: {qa["question"]}'
                 f'<span class="meta">{qa["difficulty"]}</span></div>'
                 f'<div class="a">{qa["answer"]}</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)




# ✅ placeholder: once you paste Part 2 below, safe ai image generation will work
def run_safe_ai_image_generator():
    st.info("Paste Part 2 code to enable Safe AI Image Generator.")
# main.py
import re
from io import BytesIO
import streamlit as st
from huggingface_hub import InferenceClient
import config

# add this model if the existing model is not working below --> "black-forest-labs/FLUX.1-schnell"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
ENHANCE_SYS = ("Improve prompts for text-to-image. Return ONLY the enhanced prompt. "
               "Add subject, style, lighting, camera/angle, background, colors. Keep it safe.")
NEGATIVE = "nsfw, nude, nudity, naked, erotic, porn, explicit, gore, blood, violence, weapon, hate symbols"
WORDS = ["nude","nudity","porn","sex","sexual","explicit","erotic","fetish","nsfw","blood","gore","dismember",
         "decapitate","kill","murder","suicide","self-harm","gun","weapon","knife","bomb","terror","hate",
         "racism","nazi","abuse","drugs","hate speech"]
PATS = [r"\b(nude|nudity|topless|nsfw)\b", r"\b(sex|sexual|porn|explicit|erotic|fetish)\b",
        r"\b(gore|blood|dismember|decapitat)\w*\b", r"\b(kill|murder|suicide|self[-\s]?harm)\b",
        r"\b(gun|weapon|knife|bomb|explosive)\b", r"\b(hate|racis|nazi)\w*\b"]

def is_safe(p: str):
    p2 = p.lower()
    for w in WORDS:
        if w in p2: return False, f"Blocked keyword: {w}"
    for pat in PATS:
        if re.search(pat, p, flags=re.I): return False, "Blocked unsafe pattern"
    return True, ""

img_client = InferenceClient(provider="hf-inference", api_key=config.HF_API_KEY)

def enhance_prompt(raw: str) -> str:
    from hf import generate_response
    out = generate_response(f"{ENHANCE_SYS}\nUser prompt: {raw}", temperature=0.4, max_tokens=220)
    return (out or raw).strip()

def gen_image(prompt: str):
    ok, reason = is_safe(prompt)
    if not ok: return None, f"⚠️ Prompt contains restricted/unsafe content. {reason}. Please modify and try again."
    try:
        return img_client.text_to_image(prompt=prompt, negative_prompt=NEGATIVE, model=MODEL_ID), None
    except Exception as e:
        msg = str(e)
        if "negative_prompt" in msg or "unexpected keyword" in msg:
            try: return img_client.text_to_image(prompt=prompt, model=MODEL_ID), None
            except Exception as e2: msg = str(e2)
        if any(x in msg for x in ["402", "Payment Required", "pre-paid credits"]):
            return None, "❌ Image backend requires credits or model not available on hf-inference.\n\nRaw error: " + msg
        if "404" in msg or "Not Found" in msg:
            return None, "❌ Model not served on this provider route (hf-inference).\n\nRaw error: " + msg
        return None, "Error during image generation: " + msg

def main():
    st.set_page_config(page_title="Safe AI Image Generator", layout="centered")
    st.title("🖼️ Safe AI Image Generator (Hugging Face)")
    st.info("Flow: You enter a prompt → we enhance it (HF text) → we generate the image (HF Inference).")

    with st.form("image_form"):
        raw = st.text_area("Image Description", height=120,
                           placeholder="Example: A cozy cabin in snowy mountains at sunrise, cinematic lighting")
        submit = st.form_submit_button("Generate Image")

    if submit:
        if not raw.strip(): st.warning("⚠️ Please enter an image description."); return
        with st.spinner("Enhancing your prompt..."): final_prompt = enhance_prompt(raw.strip())
        ok, reason = is_safe(final_prompt)
        if not ok: st.error(f"⚠️ Unsafe enhanced prompt. {reason}. Please rephrase and try again."); return
        st.markdown("#### Enhanced Prompt"); st.code(final_prompt)
        with st.spinner("Generating image..."): img, err = gen_image(final_prompt)
        if err: st.error(err); return
        st.image(img, caption="Generated Image", use_container_width=True)
        st.session_state.generated_image = img

    img = st.session_state.get("generated_image")
    if img:
        buf = BytesIO(); img.save(buf, format="PNG")
        st.download_button("📥 Download Image", buf.getvalue(), "ai_generated_image.png", "image/png")

if __name__ == "__main__":
    main()





def main():
    st.sidebar.title("Choose AI Feature")
    opt = st.sidebar.selectbox("", ["AI Teaching Assistant", "Math Mastermind", "Safe AI Image Generator"])
    if opt == "AI Teaching Assistant": run_ai_teaching_assistant()
    elif opt == "Math Mastermind": run_math_mastermind()
    else: run_safe_ai_image_generator()

if __name__ == "__main__":
    main()
