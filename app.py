import io
import os
import re
import hashlib
import requests
import streamlit as st
from pypdf import PdfReader

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
CHAT_MODEL = 'qwen3:1.7b'
EMBED_MODEL = 'nomic-embed-text'

st.set_page_config(page_title='ThaiLocal AI', page_icon='🍜', layout='wide')
st.title('🍜 ThaiLocal AI')
st.caption('Thai food assistant using local PDF knowledge, Ollama, and source-cited retrieval')

if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'messages' not in st.session_state:
    st.session_state.messages = []


def split_text(s, size=1100, overlap=150):
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        return []
    out = []
    step = size - overlap
    for start in range(0, len(s), step):
        out.append(s[start:start + size])
        if start + size >= len(s):
            break
    return out


def extract_text(name, payload):
    if name.lower().endswith('.pdf'):
        reader = PdfReader(io.BytesIO(payload))
        return [(i + 1, page.extract_text() or '') for i, page in enumerate(reader.pages)]
    return [(1, payload.decode('utf-8-sig', errors='replace'))]


def embedding(s):
    response = requests.post(f'{OLLAMA_URL}/api/embed', json={'model': EMBED_MODEL, 'input': s}, timeout=120)
    response.raise_for_status()
    return response.json()['embeddings'][0]


def cosine(a, b):
    import math
    return sum(x*y for x,y in zip(a,b)) / (math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)) + 1e-12)


def answer_with_ollama(question, contexts):
    evidence = '\n\n'.join(f'[{i}] Source: {c["name"]}, page {c["page"]}\n{c["text"]}' for i,c in enumerate(contexts, 1))
    prompt = ("Answer the question using ONLY the source passages below. Cite sources by filename and page "
              "in your response. If information is absent, state that the uploaded files do not establish it. "
              "Do not invent nutritional data or allergy safety. You may answer in Thai or English.\n\n"
              f"SOURCES:\n{evidence}\n\nQUESTION: {question}\nANSWER:")
    response = requests.post(f'{OLLAMA_URL}/api/generate', json={'model': CHAT_MODEL, 'prompt': prompt, 'stream': False}, timeout=240)
    response.raise_for_status()
    return response.json()['response']

with st.sidebar:
    st.header('Document collection')
    files = st.file_uploader('Add 4–5 sources about Thai food', type=['pdf', 'txt', 'md'], accept_multiple_files=True)
    if st.button('Index selected files', use_container_width=True):
        if not files:
            st.warning('Choose files first.')
        else:
            count = 0
            with st.spinner('Extracting text and generating local embeddings…'):
                try:
                    for f in files:
                        raw = f.getvalue()
                        digest = hashlib.sha256(raw).hexdigest()
                        if digest in st.session_state.documents:
                            continue
                        local_chunks = []
                        for page_no, page_text in extract_text(f.name, raw):
                            for part in split_text(page_text):
                                local_chunks.append({'name': f.name, 'page': page_no, 'text': part, 'vector': embedding(part)})
                        if not local_chunks:
                            st.warning(f'{f.name}: no searchable text found (scanned PDFs require OCR).')
                            continue
                        st.session_state.documents[digest] = {'name': f.name, 'bytes': raw}
                        st.session_state.chunks.extend(local_chunks)
                        count += 1
                    st.success(f'Indexed {count} new file(s).')
                except (requests.RequestException, ValueError, Exception) as e:
                    st.error(f'Indexing failed: {e}. Confirm Ollama is running and models are downloaded.')
    st.subheader('Your files')
    if not st.session_state.documents:
        st.write('No files indexed yet.')
    for digest, f in list(st.session_state.documents.items()):
        st.download_button(f'📄 {f["name"]}', data=f['bytes'], file_name=f['name'], key='download_' + digest)
    if st.button('Clear documents'):
        st.session_state.documents = {}
        st.session_state.chunks = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m['role']):
        st.markdown(m['content'])

question = st.chat_input('Ask about Thai dishes, regions, ingredients, or dietary preferences…')
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        if not st.session_state.chunks:
            result = 'Please upload and index your food knowledge documents first.'
        else:
            try:
                qv = embedding(question)
                ranked = sorted(st.session_state.chunks, key=lambda c: cosine(c['vector'], qv), reverse=True)[:4]
                result = answer_with_ollama(question, ranked)
                st.markdown(result)
                st.markdown('**Retrieved sources:**')
                for c in ranked:
                    st.write(f'📄 {c["name"]} — p. {c["page"]}')
            except requests.RequestException as e:
                result = f'Cannot reach Ollama: {e}. Try `ollama serve` and pull both models.'
        if not st.session_state.chunks or result.startswith('Cannot reach'):
            st.markdown(result)
    st.session_state.messages.append({'role': 'assistant', 'content': result})
