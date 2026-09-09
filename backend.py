import os
import re
import sqlite3
import uuid
import base64
import json
import hashlib
from io import BytesIO
from datetime import datetime
from typing import Optional, List, Dict, Any
# 在现有导入下方添加
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import torch
import hashlib
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ----------------------------- 导入知识库 -----------------------------
from knowledge import KNOWLEDGE_BASE as BRIEF_KB
from knowledge1 import KNOWLEDGE_BASE1 as DETAIL_KB_RAW

# ----------------------------- 类别映射（22类） -----------------------------
EN_NAMES = [
    "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous", "Candidiasis",
    "Drug Eruption", "Eczema", "Infestations/Bites", "Lichen", "Lupus",
    "Moles", "Psoriasis", "Rosacea", "Seborrheic Keratoses", "Skin Cancer",
    "Sun/Sunlight Damage", "Tinea", "Unknown/Normal", "Vascular Tumors",
    "Vasculitis", "Vitiligo", "Warts"
]

CN_NAMES = [
    "痘痘", "光化性角化病", "皮肤良性肿瘤", "大疱性皮肤病", "念珠菌病",
    "药疹", "湿疹", "虫咬/寄生虫感染", "扁平苔藓", "红斑狼疮",
    "色素痣", "银屑病", "玫瑰痤疮", "老年疣", "皮肤恶性肿瘤",
    "日光损伤/晒斑", "癣", "未识别或正常皮肤", "血管性肿瘤/畸形",
    "血管炎", "白癜风", "疣"
]

EN2CN = dict(zip(EN_NAMES, CN_NAMES))
CN2EN = dict(zip(CN_NAMES, EN_NAMES))

def convert_detail_kb():
    detail_kb = {}
    for key, value in DETAIL_KB_RAW.items():
        match = re.search(r'\(([^)]+)\)', key)
        if match:
            en_key = match.group(1)
            detail_kb[en_key] = value
        else:
            if key in EN_NAMES:
                detail_kb[key] = value
            else:
                for en, cn in EN2CN.items():
                    if cn in key:
                        detail_kb[en] = value
                        break
    return detail_kb

DETAIL_KB = convert_detail_kb()

# ----------------------------- 模型加载与推理 -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------- 千问大模型 & 向量检索 -----------------------------
QWEN_PATH = r"D:\skinDiseasesDetection-main\skinDiseasesDetection-main\Qwen-7B-Chat"
M3E_PATH = r"D:\skinDiseasesDetection-main\skinDiseasesDetection-main\m3e-small"

qwen_model = None
qwen_tokenizer = None
embedder = None
knowledge_chunks = []      # 存储文本块
knowledge_embeddings = []  # 存储对应向量

def load_qwen():
    global qwen_model, qwen_tokenizer
    try:
        qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,   # 半精度节省显存
            # load_in_8bit=True,         # 如果显存仍不足可启用
        )
        qwen_model.eval()
        print("千问模型加载成功")
    except Exception as e:
        print(f"千问模型加载失败：{e}，将使用简单问答模式")

def load_embedder():
    global embedder
    try:
        embedder = SentenceTransformer(M3E_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
        print("M3E 嵌入模型加载成功")
    except Exception as e:
        print(f"M3E 模型加载失败：{e}")

# 构建向量知识库（启动时调用一次）
def build_vector_kb():
    global knowledge_chunks, knowledge_embeddings
    if embedder is None:
        return
    chunks = []
    
    # 1. 从 knowledge.py 提取
    for en_name, info in BRIEF_KB.items():
        cn_name = EN2CN.get(en_name, en_name)
        for key, val in info.items():
            if val and val != "无":
                chunks.append(f"疾病：{cn_name} ({en_name}) - {key}：{val}")
    
    # 2. 从 knowledge1.py 提取（更详细）
    for key, info in DETAIL_KB.items():
        cn_name = EN2CN.get(key, key)
        for subkey, subval in info.items():
            if subval and subval != "无":
                chunks.append(f"疾病：{cn_name} ({key}) - {subkey}：{subval}")
    
    # 3. 从 knowledge2.json 提取问答对
    try:
        with open("knowledge2.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("knowledge_base", []):
            disease_full = item.get("disease_name", "")
            for q_group in item.get("questions", []):
                for sub in q_group.get("sub_questions", []):
                    question = sub.get("question", "")
                    answer = sub.get("answer", "")
                    if question and answer:
                        chunks.append(f"问：{question}\n答：{answer}")
    except Exception as e:
        print(f"加载 knowledge2.json 失败：{e}")
    
    # 去重（基于内容哈希）
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        h = hashlib.md5(chunk.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_chunks.append(chunk)
    
    if not unique_chunks:
        print("知识库为空")
        return
    
    # 批量计算向量（分块避免OOM）
    batch_size = 64
    embeddings = []
    for i in range(0, len(unique_chunks), batch_size):
        batch = unique_chunks[i:i+batch_size]
        emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    knowledge_embeddings = np.vstack(embeddings)
    knowledge_chunks = unique_chunks
    print(f"向量知识库构建完成，共 {len(knowledge_chunks)} 条知识")

def retrieve_relevant_chunks(query: str, top_k=3) -> List[Tuple[str, float]]:
    """检索与query最相关的知识块，返回 (chunk, score) 列表"""
    if embedder is None or len(knowledge_embeddings) == 0:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    # 余弦相似度
    scores = np.dot(knowledge_embeddings, q_emb) / (np.linalg.norm(knowledge_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [(knowledge_chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0.2]

def generate_with_qwen(question: str, context: str) -> str:
    """调用千问生成答案，context为检索到的知识"""
    if qwen_model is None or qwen_tokenizer is None:
        return None
    prompt = f"""你是一个皮肤病领域的专业医生助手。请根据以下参考知识，准确、简洁地回答用户的问题。如果参考知识不足以回答问题，请如实告知并给出通用建议。

参考知识：
{context}

用户问题：{question}

回答："""
    messages = [
        {"role": "system", "content": "你是一个皮肤病专家。"},
        {"role": "user", "content": prompt}
    ]
    # 千问-7B-Chat 使用 chat 模板
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer(text, return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=qwen_tokenizer.eos_token_id,
            pad_token_id=qwen_tokenizer.eos_token_id if qwen_tokenizer.eos_token_id is not None else qwen_tokenizer.pad_token_id,
        )
    response = qwen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()
def load_model(weight_path="best_efficientnet_b0.pth", num_classes=22):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    print(f"模型加载失败：{e}")
    model = None

def preprocess_image(image: Image.Image) -> torch.Tensor:
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img = cv2.inpaint(img, thresh, 3, cv2.INPAINT_TELEA)
    img = cv2.GaussianBlur(img, (3,3), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    h, w = img.shape[:2]
    side = min(h, w)
    start_x = (w - side) // 2
    start_y = (h - side) // 2
    img_crop = img[start_y:start_y+side, start_x:start_x+side]
    img_resized = cv2.resize(img_crop, (224, 224))
    img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_pil).unsqueeze(0).to(device)
    return tensor

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)
        weights = torch.mean(self.gradients, dim=(2,3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def generate_gradcam(input_tensor, pred_idx):
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(input_tensor, pred_idx)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_disp = input_tensor[0].cpu().detach().numpy().transpose(1,2,0)
    img_disp = img_disp * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_disp = np.clip(img_disp, 0, 1) * 255
    img_disp = img_disp.astype(np.uint8)
    superimposed = cv2.addWeighted(img_disp, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode('.png', superimposed)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return b64_str

# ----------------------------- 数据库 -----------------------------
DB_PATH = "history.db"
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            password TEXT NOT NULL,
            create_time TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            create_time TEXT NOT NULL,
            name TEXT,
            age INTEGER,
            gender TEXT,
            image_path TEXT,
            top3_result TEXT,
            detail_report TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def generate_user_id():
    today = datetime.now().strftime("%Y%m%d")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    prefix = f"P{today}-"
    c.execute("SELECT user_id FROM users WHERE user_id LIKE ?", (prefix + '%',))
    rows = c.fetchall()
    conn.close()
    max_seq = 0
    for (uid,) in rows:
        seq_str = uid.split('-')[-1]
        if seq_str.isdigit():
            max_seq = max(max_seq, int(seq_str))
    new_seq = max_seq + 1
    return f"{prefix}{new_seq:03d}"

def save_detection(user_id, name, age, gender, image_path, top3_result, detail_report):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO detections (user_id, create_time, name, age, gender, image_path, top3_result, detail_report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, now, name, age, gender, image_path, top3_result, detail_report))
    conn.commit()
    conn.close()

def get_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT create_time, top3_result, detail_report FROM detections
        WHERE user_id = ? ORDER BY create_time DESC
    ''', (user_id,))
    rows = c.fetchall()
    conn.close()
    history = []
    for row in rows:
        create_time, top3_json, detail_report = row
        top3_list = json.loads(top3_json)
        top1 = top3_list[0] if top3_list else {}
        history.append({
            "time": create_time,
            "disease": top1.get("disease_cn", "未知"),
            "confidence": f"{top1.get('score', 0)*100:.1f}%",
            "detail_report": detail_report
        })
    return history

def answer_question(q: str) -> str:
    q_lower = q.lower()
    matched_disease = None
    for en, cn in EN2CN.items():
        if cn in q or en.lower() in q_lower:
            matched_disease = en
            break
    if not matched_disease:
        return "请问您想了解哪种皮肤病？可以输入疾病名称（如痤疮、湿疹、银屑病等）或具体问题。"
    detail = DETAIL_KB.get(matched_disease, BRIEF_KB.get(matched_disease, {}))
    if not detail:
        return f"未找到“{EN2CN[matched_disease]}”的详细资料。"
    if any(k in q_lower for k in ["病因", "原因", "怎么引起"]):
        return detail.get("病因", "暂无病因描述。")
    elif any(k in q_lower for k in ["症状", "表现", "什么样"]):
        return detail.get("症状", "暂无症状描述。")
    elif any(k in q_lower for k in ["用药", "药物", "吃什么药", "涂什么药"]):
        return detail.get("用药", "暂无用药建议。")
    elif any(k in q_lower for k in ["建议", "注意", "护理", "日常"]):
        return detail.get("建议", "暂无日常建议。")
    else:
        return detail.get("简介", "暂无简介。")

# ----------------------------- FastAPI 应用 -----------------------------
app = FastAPI(title="皮肤病智能检测系统")

@app.post("/api/register")
async def register(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    password: str = Form(...)
):
    user_id = generate_user_id()
    hashed = hash_password(password)
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (user_id, name, age, gender, password, create_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, name, age, gender, hashed, create_time))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="用户ID重复，请重试")
    finally:
        conn.close()
    return {
        "user_id": user_id,
        "name": name,
        "age": age,
        "gender": gender,
        "message": "注册成功"
    }

@app.post("/api/login")
async def login(
    user_id: str = Form(...),
    password: str = Form(...)
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, age, gender, password FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="用户编号不存在")
    name, age, gender, hashed = row
    if hash_password(password) != hashed:
        raise HTTPException(status_code=401, detail="密码错误")
    return {
        "user_id": user_id,
        "name": name,
        "age": age,
        "gender": gender,
        "message": "登录成功"
    }

@app.post("/api/skin_detect")
async def skin_detect(
    user_id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    file: UploadFile = File(...)
):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    contents = await file.read()
    pil_img = Image.open(BytesIO(contents)).convert("RGB")
    input_tensor = preprocess_image(pil_img)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = []
    for idx in top3_idx:
        en = EN_NAMES[idx]
        cn = EN2CN[en]
        score = float(probs[idx])
        top3.append({"disease_en": en, "disease_cn": cn, "score": score})
    pred_idx = top3_idx[0]
    gradcam_b64 = generate_gradcam(input_tensor, pred_idx)
    top1_en = top3[0]["disease_en"]
    brief_info = BRIEF_KB.get(top1_en, {})
    detail_info = DETAIL_KB.get(top1_en, {})
    detail_report = f"""
========================================
        皮肤病智能检测报告
========================================
患者信息：{name}（{gender}，{age}岁）
检测时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
用户编号：{user_id}

【Top-3 预测结果】
1. {top3[0]['disease_cn']}（{top3[0]['disease_en']}）置信度：{top3[0]['score']*100:.1f}%
2. {top3[1]['disease_cn']}（{top3[1]['disease_en']}）置信度：{top3[1]['score']*100:.1f}%
3. {top3[2]['disease_cn']}（{top3[2]['disease_en']}）置信度：{top3[2]['score']*100:.1f}%

【详细疾病知识 - {top3[0]['disease_cn']}】
简介：{detail_info.get('简介', '无')}
症状：{detail_info.get('症状', '无')}
病因：{detail_info.get('病因', '无')}
用药：{detail_info.get('用药', '无')}
建议：{detail_info.get('建议', '无')}

温馨提示：
1. 本结果为AI辅助分析，仅供健康参考，不构成医疗诊断意见。
2. 皮肤病病因复杂，需结合病史、面诊及实验室检查确诊。
3. 如病情较重、反复发作或伴随全身不适，请及时前往医院皮肤科就诊，切勿自行用药。
========================================
"""
    img_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    img_path = os.path.join(UPLOAD_DIR, img_filename)
    pil_img.save(img_path)
    top3_json = json.dumps(top3, ensure_ascii=False)
    save_detection(user_id, name, age, gender, img_path, top3_json, detail_report)
    return JSONResponse(content={
        "top3": top3,
        "gradcam_base64": gradcam_b64,
        "brief_info": brief_info,
        "detail_report": detail_report
    })

@app.get("/api/history")
async def history(user_id: str):
    records = get_history(user_id)
    return {"history": records}

# 加载 knowledge2.json 并构建映射
KNOWLEDGE2_PATH = "knowledge2.json"
QUESTION_ANSWER_MAP = {}

def load_knowledge2():
    global QUESTION_ANSWER_MAP
    try:
        with open(KNOWLEDGE2_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("knowledge_base", []):
            for q_group in item.get("questions", []):
                for sub in q_group.get("sub_questions", []):
                    question = sub.get("question", "")
                    answer = sub.get("answer", "")
                    if question and answer:
                        QUESTION_ANSWER_MAP[question] = answer
        print(f"加载 knowledge2 问答库，共 {len(QUESTION_ANSWER_MAP)} 条")
    except Exception as e:
        print(f"加载 knowledge2.json 失败: {e}")

load_knowledge2()

@app.get("/api/chat")
async def chat(q: str):
    # 如果千问模型可用，使用 RAG 生成
    if qwen_model is not None and embedder is not None and knowledge_embeddings is not None:
        try:
            # 检索相关知识点
            relevant = retrieve_relevant_chunks(q, top_k=4)
            if relevant:
                context = "\n\n".join([chunk for chunk, _ in relevant])
            else:
                context = "未找到直接相关的知识。"
            answer = generate_with_qwen(q, context)
            if answer:
                return {"answer": answer}
        except Exception as e:
            print(f"千问生成失败：{e}，降级到简单问答")
    # fallback：原来的简单问答
    answer = answer_question(q)
    return {"answer": answer}

@app.get("/api/get_knowledge2")
async def get_knowledge2():
    try:
        with open("knowledge2.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)