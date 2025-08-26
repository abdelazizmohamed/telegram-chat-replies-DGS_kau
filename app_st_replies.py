import os
import gdown
import zipfile
import streamlit as st
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict, deque
import json

# -------------------- تحميل البيانات من Google Drive --------------------
file_url = "https://drive.google.com/uc?id=1CMlkOVj4pv9VxCLhoM5GNgivbt5Jl7Bu"  # استبدل بـ رابط Google Drive المباشر
output_path = "telegram-chat-replies-DGS_kau.zip"   # تحديد مكان حفظ الملف بعد تحميله

# تحميل الملف باستخدام gdown
try:
    gdown.download(file_url, output_path, quiet=False)
    st.success("تم تحميل البيانات بنجاح!")
except Exception as e:
    st.error(f"فشل تحميل البيانات: {e}")

# التحقق من وجود الملف قبل فك الضغط
if os.path.exists(output_path):
    # فك ضغط الملف إذا كان مضغوطًا
    zip_file_path = output_path  # الملف المضغوط
    extract_folder = "data_folder"  # المجلد الذي سيتم استخراج الملفات فيه

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    st.success("تم فك ضغط البيانات بنجاح!")
else:
    st.error("الملف المضغوط غير موجود.")

# -------------------- استخراج الرسائل من HTML --------------------
def parse_telegram_html(html_path: str):
    html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    msgs = []

    for msg_div in soup.select("div.message"):
        mid = msg_div.get("id") or ""
        user_display = ""
        username = ""
        from_name = msg_div.select_one(".from_name")
        if from_name:
            user_display = from_name.get_text(" ", strip=True)
            a = from_name.find("a", href=True)
            if a and "t.me" in a["href"]:
                handle = a["href"].rstrip("/").split("/")[-1]
                username = "@" + handle if handle else ""
            if not username:
                m = re.search(r"\(@([A-Za-z0-9_]+)\)", user_display)
                if m:
                    username = f"@{m.group(1)}"
        if not username:
            a_any = msg_div.select_one('a[href*="t.me"]')
            if a_any and a_any.has_attr("href"):
                handle = a_any["href"].rstrip("/").split("/")[-1]
                if handle and not handle.startswith(("joinchat", "+")):
                    username = "@" + handle if not handle.startswith("@") else handle
        date_div = msg_div.select_one(".date")
        date_str = date_div.get_text(" ", strip=True) if date_div else ""
        try:
            msg_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except:
            msg_date = None
        text_div = msg_div.select_one(".text")
        message = text_div.get_text(" ", strip=True) if text_div else ""
        reply_to = ""
        reply_anchor = msg_div.select_one(".reply_to a[href]")
        if reply_anchor:
            href = reply_anchor["href"]
            m = re.search(r"go_to_message(\d+)", href)
            if m:
                reply_to = f"message{m.group(1)}"
        if user_display or message:
            msgs.append({
                "id": mid,
                "user": user_display or "",
                "username": username or "",
                "date": date_str,
                "datetime": msg_date,
                "message": message,
                "reply_to": reply_to,
            })
    return msgs

# -------------------- Embeddings / FAISS --------------------
def embed_texts(model, texts, batch_size=64, show_tqdm=True):
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        show_progress_bar=show_tqdm, normalize_embeddings=True)
    return vecs.astype("float32")

def save_index(out_dir, index, metas, model_name):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(outp / "index.faiss"))
    meta = {"model": model_name, "count": len(metas)}
    (outp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (outp / "rows.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in metas), encoding="utf-8"
    )

def load_index(out_dir):
    outp = Path(out_dir)
    if os.path.exists(str(outp / "index.faiss")):
        index = faiss.read_index(str(outp / "index.faiss"))
        metas = [json.loads(l) for l in (outp / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        meta = json.loads((outp / "meta.json").read_text(encoding="utf-8"))
        return index, metas, meta
    else:
        st.error("ملف index.faiss غير موجود!")
        return None, None, None

# -------------------- Answers --------------------
def _build_children_map(metas):
    children = defaultdict(list)
    id_to_idx = {}
    for i, m in enumerate(metas):
        mid = m.get("id","")
        if mid:
            id_to_idx[mid] = i
    for i, m in enumerate(metas):
        parent = m.get("reply_to","")
        if parent:
            children[parent].append(i)
    return children, id_to_idx

def cmd_answers(out_dir, q, k=5, max_replies=20, max_depth=5):
    index, metas, meta = load_index(out_dir)
    if index is None:
        return []
    
    model_name = meta.get("model", "distiluse-base-multilingual-cased-v2")
    model = SentenceTransformer(model_name)
    children, id_to_idx = _build_children_map(metas)
    q_vec = embed_texts(model, [q], batch_size=1, show_tqdm=False)
    D, I = index.search(q_vec, k)
    results = []
    for idx in I[0]:
        seed = metas[int(idx)]
        sid = seed.get("id","")
        queue = deque((c,1) for c in children.get(sid,[]))
        replies = []
        printed = 0
        while queue:
            r_idx, depth = queue.popleft()
            if depth > max_depth: continue
            r = metas[r_idx]
            replies.append((depth, r))
            printed += 1
            if printed >= max_replies: break
            for cc in children.get(r.get("id",""), []):
                queue.append((cc, depth+1))
        results.append({"seed": seed, "replies": replies})
    return results

# -------------------- واجهة Streamlit بالعربية وRTL --------------------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
/* تأكيد الاتجاه من اليمين لليسار */
body { direction: rtl; font-family: 'Cairo', sans-serif !important; }
h1.title { font-family: 'Cairo', sans-serif !important; font-size: 2.5em; text-align: center; color: #1A1A1A; }
p, span, div, h2, h3, h4 { font-family: 'Cairo', sans-serif !important; }
</style>
<h1 class="title">ابحث في شات قروب التلقرام (الدراسات العليا جامعة الملك عبدالعزيز)</h1>
""", unsafe_allow_html=True)

# -------------------- واجهة البحث --------------------
out_dir = st.text_input("المجلد الذي يحتوي على الفهرس (Index folder)", "./data_folder")
query_text = st.text_input("أدخل نص البحث", "امتى يبدأ التقديم؟")
k = st.number_input("عدد الرسائل الأساسية المراد عرضها (Top-k)", min_value=1, value=5)
max_replies = st.number_input("الحد الأقصى للردود لكل رسالة", min_value=1, value=20)
max_depth = st.number_input("الحد الأقصى لعمق الردود", min_value=1, value=5)
show_only_with_replies = st.checkbox("عرض الرسائل التي لها ردود فقط")

if st.button("عرض الردود"):
    all_results = cmd_answers(out_dir, query_text, k=k, max_replies=max_replies, max_depth=max_depth)

    if show_only_with_replies:
        all_results = [r for r in all_results if len(r['replies'])>0]

    # عرض الرسائل والردود
    for res in all_results:
        seed = res['seed']
        st.markdown(f"**{seed['date']} | {seed['user']} ({seed['username']})**")
        st.write(seed['message'])

        if res['replies']:
            with st.expander(f"عرض {len(res['replies'])} رد"):
                for depth, r in res['replies']:
                    indent = " " * (depth*4)
                    st.markdown(f"{indent}- {r['date']} | {r['user']} ({r['username']})")
                    st.write(f"{indent}  {r['message']}")
        st.markdown("---")
