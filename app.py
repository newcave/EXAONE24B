import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ 모델명 및 로컬 캐시 설정
MODEL_NAME = "LGAI-EXAONE/EXAONE-Deep-2.4B"

@st.cache_resource  # ✅ Streamlit 캐싱 (최초 실행 후 유지)
def load_model():
    """EXAONE 모델 및 토크나이저 로드 (최초 실행 시만 로드)"""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, revision="main"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # GPU 최적화
        device_map="auto",  # GPU 자동 할당
        trust_remote_code=True,
        revision="main"
    )
    model.eval()
    return tokenizer, model

# ✅ 모델 로드 (Streamlit 캐싱 적용)
tokenizer, model = load_model()

# ✅ 응답 생성 함수
def generate_response(prompt):
    """EXAONE 모델이 답변을 생성"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1000,  # 너무 길면 실행 시간 초과 가능
            do_sample=False,  # 랜덤성 제거
            temperature=0.1,  # 창의성 최소화
            top_p=0.9,  # 핵심 정보 위주
            repetition_penalty=1.2  # 반복 방지
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Streamlit UI 구성
st.title("🧠 EXAONE-Deep-2.4B Chatbot")
st.write("💬 **Fact 기반 AI 챗봇**입니다. 질문을 입력하세요.")

# 사용자 입력 받기
user_input = st.text_input("질문을 입력하세요:", "")

# 질문이 입력되었을 때 응답 생성
if user_input:
    with st.spinner("EXAONE이 답변을 생성 중..."):
        response = generate_response(user_input)
    st.markdown("### 🤖 EXAONE의 답변:")
    st.write(response)
