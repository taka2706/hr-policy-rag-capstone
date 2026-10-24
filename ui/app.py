"""Streamlit UI. Run: streamlit run ui/app.py   (API must be up: make api)"""
import json, os
import httpx
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="HR Policy Assistant")
st.title("HR Policy Assistant")

try:
    h = httpx.get(f"{API}/health", timeout=3).json()
    st.caption(f"API v{h['api_version']} · backend={h['backend']} · {h['chunks']} chunks")
except Exception:
    st.error(f"API not reachable at {API}. Start it with `make api`.")
    st.stop()

q = st.text_input("Ask a policy question", max_chars=1000)
if st.button("Ask") and q.strip():
    box, text, final = st.empty(), "", None
    try:
        with httpx.stream("POST", f"{API}/ask", json={"question": q}, timeout=60) as r:
            if r.status_code != 200:
                st.error(f"{r.status_code}: {r.read().decode()[:300]}")
                st.stop()
            for line in r.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                if ev["type"] == "token":
                    text += ev["data"]; box.markdown(text + "▌")
                elif ev["type"] == "final":
                    final = ev["data"]
                elif ev["type"] == "error":
                    st.error(ev["data"])
        box.markdown(text)
        if final:
            if final["refused"]:
                st.warning("No supporting policy text found.")
            elif final["citations"]:
                st.caption("Sources: " + ", ".join(final["citations"]))
    except httpx.HTTPError as e:
        st.error(f"Request failed: {e}")
