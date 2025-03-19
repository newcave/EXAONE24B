import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ 모델 저장 경로 및 Hugging Face 모델명
MODEL_NAME = "LGAI-EXAONE/EXAONE-Deep-2.4B"

@st.cache_resource  # 모델 캐싱하여 로드 속도 향상
def load_model():
    """모델과 토크나이저를 로드"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

# 모델 로드
tokenizer, model = load_model()

# ✅ 응답 생성 함수
def generate_response(prompt):
    """EXAONE 모델이 답변을 생성"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.2
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Streamlit UI 구성
st.title("🧠 EXAONE-Deep-2.4B Chatbot")
st.write("💬 **Fact 기반 AI 챗봇**입니다. 질문을 입력하세요.")

# 채팅 입력창
user_input = st.text_input("질문을 입력하세요:", "")

# 질문이 입력되었을 때 응답 생성
if user_input:
    with st.spinner("EXAONE이 답변을 생성 중..."):
        response = generate_response(user_input)
    st.markdown(f"### 🤖 EXAONE의 답변:")
    st.write(response)
