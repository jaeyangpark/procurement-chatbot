import os
import streamlit as st
import time
from rag_chain import get_qa_chain, load_and_embed_pdfs

st.set_page_config(page_title="조달청 AI 비서", layout="wide")
st.title("🧾 조달청 관련 질문 RAG 시스템")

# API 키 환경변수 설정 (Streamlit Cloud에서는 Secrets에 저장 권장)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 조달청 관련 궁금하신 점을 물어보세요."}]

# PDF 임베딩 버튼
if st.button("🔁 PDF 임베딩 새로하기"):
    with st.spinner("📄 PDF 임베딩 중입니다..."):
        load_and_embed_pdfs("data")
    st.success("✅ 임베딩 완료!")

# 대화 이력 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 질문 입력
if prompt := st.chat_input("조달청 관련 질문을 입력해주세요:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    qa_chain = get_qa_chain()
    with st.chat_message("assistant"):
        with st.spinner("🤖 답변 생성 중입니다..."):
            result = qa_chain({"query": prompt})
        response = result["result"]
        source_docs = result.get("source_documents", [])

        # 스트리밍 출력
        placeholder = st.empty()
        full_response = ""
        for chunk in response.split():
            full_response += chunk + " "
            placeholder.markdown(full_response + "▌")
            time.sleep(0.03)
        placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # 관련 문서 출력
        if source_docs:
            with st.expander("📎 관련 문서 보기"):
                for doc in source_docs:
                    st.markdown(f"- {doc.metadata['source']}:\n```{doc.page_content[:300]}...```")
