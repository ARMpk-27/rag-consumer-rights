"""Streamlit App: ผู้ช่วยสิทธิผู้บริโภคและการซื้อของออนไลน์ (RAG)"""
import streamlit as st

from rag import (
    NOT_FOUND_MESSAGE, build_chunks, build_index, build_messages, clean_answer,
    embed_passages, load_documents, retrieval_query, search,
)

st.set_page_config(page_title="ผู้ช่วยสิทธิผู้บริโภค", page_icon="🛒", layout="centered")

DATA_DIR = "data"
EMBED_MODEL = "intfloat/multilingual-e5-small"            # โมเดลเล็ก รองรับภาษาไทย
LLM_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]  # Production models บน Groq (ต.ค. 2569)
EXAMPLES = [
    "สั่งของเก็บเงินปลายทางแล้วได้ของไม่ตรงปก ทำอย่างไรได้บ้าง",
    "โอนเงินซื้อของจากเพจแล้วโดนบล็อก ต้องทำอย่างไร",
    "สิทธิของผู้บริโภคตามกฎหมายมีอะไรบ้าง",
    "ร้องเรียน สคบ. ได้ช่องทางไหนบ้าง",
]


@st.cache_resource(show_spinner="กำลังโหลดโมเดลและสร้างดัชนีเอกสาร ทำครั้งแรกครั้งเดียว...")
def load_rag():
    """โหลดโมเดล embedding และสร้าง FAISS index เพียงครั้งเดียวต่อเซิร์ฟเวอร์"""
    from sentence_transformers import SentenceTransformer

    docs = load_documents(DATA_DIR)
    chunks = build_chunks(docs)
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    index = build_index(embed_passages(model, chunks))
    return model, index, chunks, docs


@st.cache_resource
def get_client(api_key: str):
    from groq import Groq

    return Groq(api_key=api_key)


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def render_sources(sources: list[dict]):
    with st.expander(f"เอกสารอ้างอิงที่ใช้ตอบ ({len(sources)} ส่วน)"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['title']}**  \nหัวข้อ: {s['section']}  \nไฟล์ `{s['file']}`, คะแนนความใกล้เคียง {s['score']:.3f}")
            snippet = s["text"] if len(s["text"]) <= 350 else s["text"][:350] + "..."
            st.caption(snippet)
            if s["url"]:
                st.markdown(f"[เปิดแหล่งที่มา]({s['url']})")


def ask_llm(api_key: str, model_name: str, messages: list[dict]) -> str:
    client = get_client(api_key)
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.2)
    return clean_answer(resp.choices[0].message.content)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("ตั้งค่า")
    model_name = st.selectbox("โมเดลภาษา (Groq)", LLM_MODELS)
    top_k = st.slider("จำนวนส่วนเอกสารที่ค้นมาใช้ตอบ", 2, 8, 4)
    if st.button("เริ่มบทสนทนาใหม่", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("ข้อมูลจากเว็บไซต์ สคบ. และประกาศที่เกี่ยวข้อง ตรวจสอบเมื่อ 5 ต.ค. 2569 "
               "คำตอบเป็นข้อมูลทั่วไป ไม่ใช่คำปรึกษาทางกฎหมาย หากมีปัญหาจริง โทรสายด่วน สคบ. 1166")

# ---------- Main ----------
st.title("ถามเรื่องสิทธิผู้บริโภคและการซื้อของออนไลน์")
st.caption("ตอบจากเอกสารในคลังความรู้เท่านั้น ทุกคำตอบแสดงเอกสารที่ใช้อ้างอิง ถ้าเอกสารไม่มีคำตอบ ระบบจะบอกว่าไม่พบข้อมูล")

api_key = get_api_key()
if not api_key:
    st.error("ยังไม่ได้ตั้งค่า GROQ_API_KEY ให้ใส่คีย์ใน Settings > Secrets ของ Streamlit Cloud "
             "หรือในไฟล์ .streamlit/secrets.toml เมื่อรันในเครื่อง")
    st.stop()

model, index, chunks, docs = load_rag()
with st.sidebar:
    st.caption(f"คลังความรู้: {len(docs)} เอกสาร แบ่งเป็น {len(chunks)} ส่วน")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"])

if not st.session_state.messages:
    st.markdown("**ลองถาม:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}", width="stretch"):
            st.session_state.pending = q
            st.rerun()

question = st.chat_input("พิมพ์คำถามของคุณ") or st.session_state.pop("pending", None)

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นเอกสารและเรียบเรียงคำตอบ..."):
            results = search(model, index, chunks, retrieval_query(question, history), k=top_k)
            sources = [{"n": n, "score": score, **c.to_dict()} for n, (c, score) in enumerate(results, 1)]
            try:
                answer = ask_llm(api_key, model_name, build_messages(question, results, history))
            except Exception as e:  # แสดงสาเหตุให้ผู้ใช้เห็น แทนที่แอปจะล่ม
                answer = f"เรียกใช้โมเดลภาษาไม่สำเร็จ ({type(e).__name__}) ลองเปลี่ยนโมเดลในแถบด้านซ้ายหรือลองใหม่อีกครั้ง"
        st.markdown(answer)
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
