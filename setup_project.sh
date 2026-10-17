#!/usr/bin/env bash
# ============================================================
#  LLM From Scratch — Project Setup Script
#  Run: bash setup_project.sh
#  This will create the full folder structure + all files.
# ============================================================

set -e
PROJECT="llm-from-scratch"
mkdir -p $PROJECT/{routes,models,utils,templates}
cd $PROJECT
echo "📁 Created folder structure"

# ── app.py ──────────────────────────────────────────────────
cat > app.py << 'EOF'
from flask import Flask
from routes.main import main_bp
from routes.api  import api_bp

def create_app():
    app = Flask(__name__)
    app.secret_key = "llm-tutorial-secret"
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=7860, debug=False)
EOF

# ── requirements.txt ────────────────────────────────────────
cat > requirements.txt << 'EOF'
flask>=3.0.0
torch>=2.1.0
numpy>=1.24.0
gunicorn>=21.2.0
EOF

# ── Dockerfile ──────────────────────────────────────────────
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
RUN useradd -m -u 1000 hfuser && chown -R hfuser /app
USER hfuser
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:create_app()"]
EOF

# ── README.md ───────────────────────────────────────────────
cat > README.md << 'EOF'
---
title: llm-from-scratch
colorFrom: blue
colorTo: indigo
sdk: docker
---

# 🧠 LLM From Scratch

An interactive Flask + PyTorch dashboard walking through every stage of
training a GPT-style language model: tokenization, architecture, live
training, attention visualisation, and text generation.

## Quick Start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

## Docker

```bash
docker build -t llm-from-scratch .
docker run -p 7860:7860 llm-from-scratch
```

## Project Structure

```
llm-from-scratch/
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   └── tiny_gpt.py
├── routes/
│   ├── __init__.py
│   ├── main.py
│   └── api.py
├── utils/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── trainer.py
│   └── data_utils.py
└── templates/
    ├── base.html
    ├── index.html
    ├── tokenize.html
    ├── architecture.html
    ├── train.html
    ├── attention.html
    └── generate.html
```

## License
MIT — Educational use only.
EOF

# ── routes/__init__.py ───────────────────────────────────────
touch routes/__init__.py

# ── routes/main.py ──────────────────────────────────────────
cat > routes/main.py << 'EOF'
from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index():
    return render_template("index.html")

@main_bp.route("/tokenize")
def tokenize():
    return render_template("tokenize.html")

@main_bp.route("/architecture")
def architecture():
    return render_template("architecture.html")

@main_bp.route("/train")
def train():
    return render_template("train.html")

@main_bp.route("/attention")
def attention():
    return render_template("attention.html")

@main_bp.route("/generate")
def generate():
    return render_template("generate.html")
EOF

# ── routes/api.py ───────────────────────────────────────────
cat > routes/api.py << 'EOF'
import threading
from flask import Blueprint, request, jsonify
import torch
from models.tiny_gpt import TinyGPT
from utils.tokenizer import CharTokenizer
from utils.trainer import TrainingState, get_batch, train_loop
from utils.data_utils import SAMPLE_TEXT, BPE_EXPLAIN_STEPS

api_bp    = Blueprint("api", __name__)
t_state   = TrainingState()
tokenizer = CharTokenizer()
model_ref = {"model": None}


@api_bp.route("/tokenize", methods=["POST"])
def api_tokenize():
    text  = request.json.get("text", "")[:2000]
    tok   = CharTokenizer(text)
    ids   = tok.encode(text)
    pairs = [{"char": c, "id": i} for c, i in zip(text, ids)]
    return jsonify({
        "vocab_size": tok.vocab_size,
        "num_tokens": len(ids),
        "pairs":      pairs[:200],
        "vocab":      list(tok.char2idx.keys()),
        "bpe_steps":  BPE_EXPLAIN_STEPS
    })


@api_bp.route("/model_stats", methods=["POST"])
def model_stats():
    cfg        = request.json
    d_model    = int(cfg.get("d_model",  128))
    n_heads    = int(cfg.get("n_heads",  4))
    n_layers   = int(cfg.get("n_layers", 4))
    max_len    = int(cfg.get("max_len",  256))
    vocab_size = int(cfg.get("vocab_size", 65))
    d_ff       = d_model * 4
    m          = TinyGPT(vocab_size, d_model, n_heads, n_layers, d_ff, max_len)
    return jsonify({
        "total_params":          m.count_params(),
        "embedding_params":      vocab_size * d_model + max_len * d_model,
        "attn_params_per_layer": 4 * d_model * d_model,
        "ff_params_per_layer":   2 * d_model * d_ff,
        "d_ff":                  d_ff,
        "heads_valid":           (d_model % n_heads == 0),
        "size_mb":               round(m.count_params() * 4 / 1e6, 2)
    })


@api_bp.route("/start_training", methods=["POST"])
def start_training():
    global tokenizer, model_ref
    if t_state.is_running:
        return jsonify({"error": "Already training"}), 400
    cfg      = request.json or {}
    d_model  = int(cfg.get("d_model",  64))
    n_heads  = int(cfg.get("n_heads",  4))
    n_layers = int(cfg.get("n_layers", 2))
    steps    = min(int(cfg.get("steps", 500)), 2000)
    lr       = float(cfg.get("lr", 3e-4))
    batch    = int(cfg.get("batch_size", 32))
    block    = int(cfg.get("block_size", 64))

    tokenizer = CharTokenizer(SAMPLE_TEXT)
    data_t    = torch.tensor(tokenizer.encode(SAMPLE_TEXT), dtype=torch.long)
    gpt       = TinyGPT(tokenizer.vocab_size, d_model, n_heads, n_layers, d_model * 4, block)
    model_ref["model"] = gpt
    t_state.reset()

    train_cfg = {"lr": lr, "steps": steps, "batch_size": batch, "block_size": block}
    threading.Thread(target=train_loop, args=(gpt, data_t, train_cfg, t_state), daemon=True).start()
    return jsonify({"status": "started", "total_steps": steps, "vocab_size": tokenizer.vocab_size})


@api_bp.route("/training_status")
def training_status():
    with t_state.lock:
        losses = list(t_state.losses)
    return jsonify({
        "step": t_state.step, "total_steps": t_state.total_steps,
        "losses": losses, "status": t_state.status, "is_running": t_state.is_running
    })


@api_bp.route("/stop_training", methods=["POST"])
def stop_training():
    t_state.is_running = False
    return jsonify({"status": "stopping"})


@api_bp.route("/attention", methods=["POST"])
def get_attention():
    gpt = model_ref.get("model")
    if gpt is None:
        return jsonify({"error": "No trained model yet — run training first"}), 400
    text = request.json.get("text", "Hello world")[:64]
    ids  = tokenizer.encode(text)
    if len(ids) < 2:
        return jsonify({"error": "Enter at least 2 characters"}), 400
    gpt.eval()
    with torch.no_grad():
        gpt(torch.tensor([ids], dtype=torch.long))
    layer = max(0, min(int(request.json.get("layer", 0)), len(gpt.blocks) - 1))
    w     = gpt.get_attention_weights(layer)
    return jsonify({"tokens": list(text), "heads": w[0].cpu().tolist(), "n_heads": w.shape[1]})


@api_bp.route("/generate", methods=["POST"])
def generate():
    gpt = model_ref.get("model")
    if gpt is None:
        return jsonify({"error": "No trained model yet — run training first"}), 400
    prompt      = request.json.get("prompt", " ")[:64] or " "
    max_tokens  = min(int(request.json.get("max_tokens", 200)), 500)
    temperature = float(request.json.get("temperature", 1.0))
    top_k       = request.json.get("top_k")
    top_k       = int(top_k) if top_k else None
    ids         = tokenizer.encode(prompt)
    gpt.eval()
    with torch.no_grad():
        out = gpt.generate(torch.tensor([ids], dtype=torch.long), max_tokens, temperature, top_k)
    generated = tokenizer.decode(out[0].tolist()[len(ids):])
    return jsonify({"prompt": prompt, "generated": generated, "full": prompt + generated})
EOF

# ── models/__init__.py ───────────────────────────────────────
touch models/__init__.py

# ── models/tiny_gpt.py ──────────────────────────────────────
cat > models/tiny_gpt.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.qkv       = nn.Linear(d_model, 3 * d_model)
        self.proj      = nn.Linear(d_model, d_model)
        self.drop      = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        self.last_attn = attn.detach()
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff   = FeedForward(d_model, d_ff, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4,
                 n_layers=4, d_ff=512, max_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
        self.drop    = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        pos    = torch.arange(T, device=x.device)
        x      = self.drop(self.token_emb(x) + self.pos_emb(pos))
        mask   = torch.tril(torch.ones(T, T, device=x.device))
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.head(self.ln_f(x))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=100, temperature=1.0, top_k=None):
        for _ in range(max_new):
            ctx = idx[:, -self.max_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_attention_weights(self, layer_idx=0):
        return self.blocks[layer_idx].attn.last_attn
EOF

# ── utils/__init__.py ────────────────────────────────────────
touch utils/__init__.py

# ── utils/tokenizer.py ──────────────────────────────────────
cat > utils/tokenizer.py << 'EOF'
class CharTokenizer:
    def __init__(self, text=None):
        self.char2idx   = {}
        self.idx2char   = {}
        self.vocab_size = 0
        if text:
            self.fit(text)

    def fit(self, text):
        chars = sorted(set(text))
        self.char2idx   = {c: i for i, c in enumerate(chars)}
        self.idx2char   = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(chars)
        return self

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.idx2char.get(i, "?") for i in ids)
EOF

# ── utils/trainer.py ────────────────────────────────────────
cat > utils/trainer.py << 'EOF'
import threading
import time
import torch


class TrainingState:
    def __init__(self):
        self.lock        = threading.Lock()
        self.losses      = []
        self.step        = 0
        self.total_steps = 0
        self.is_running  = False
        self.status      = "idle"

    def reset(self):
        with self.lock:
            self.losses      = []
            self.step        = 0
            self.is_running  = True
            self.status      = "training"


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i + block_size]         for i in ix])
    y  = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def train_loop(model, data, cfg, state):
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"])
    model.train()
    state.total_steps = cfg["steps"]

    for step in range(cfg["steps"]):
        if not state.is_running:
            state.status = "stopped"
            return
        x, y = get_batch(data, cfg["block_size"], cfg["batch_size"])
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with state.lock:
            state.step = step + 1
            state.losses.append(round(loss.item(), 4))
        time.sleep(0.005)

    state.is_running = False
    state.status     = "done"
EOF

# ── utils/data_utils.py ─────────────────────────────────────
cat > utils/data_utils.py << 'EOF'
SAMPLE_TEXT = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
""" * 6


BPE_EXPLAIN_STEPS = [
    {"step": 1, "title": "Start with characters",    "desc": "Split the corpus into individual characters. e.g. 'hello' -> ['h','e','l','l','o']"},
    {"step": 2, "title": "Count adjacent pairs",     "desc": "Count every consecutive pair: ('h','e')=1, ('e','l')=1, ('l','l')=1, ('l','o')=1"},
    {"step": 3, "title": "Merge most frequent pair", "desc": "If ('l','l') appears most, merge into 'll'. Repeat across entire corpus."},
    {"step": 4, "title": "Update vocabulary",        "desc": "Add 'll' as a new token. Every 'l l' in data becomes the single token 'll'."},
    {"step": 5, "title": "Repeat until target",      "desc": "Keep merging the most frequent pairs until reaching desired vocab size (e.g. 32k, 50k)."}
]
EOF

# ── templates/base.html ─────────────────────────────────────
cat > templates/base.html << 'EOF'
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LLM From Scratch</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>
<style>
:root{--accent:#6366f1}
body{background:#0f0f1a;color:#e2e8f0;font-family:'Segoe UI',sans-serif}
.navbar{background:linear-gradient(90deg,#1e1b4b,#1e3a5f)!important;border-bottom:1px solid #334155}
.navbar-brand{font-weight:800;color:#818cf8!important}
.nav-link{color:#94a3b8!important}.nav-link:hover,.nav-link.active{color:#a5b4fc!important}
.hero{background:linear-gradient(135deg,#1e1b4b,#0f172a,#1e3a5f);border-bottom:1px solid #334155;padding:3rem 0}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px}
.btn-accent{background:var(--accent);border:none;color:#fff}.btn-accent:hover{background:#4f46e5;color:#fff}
.code-pill{background:#0f172a;border:1px solid #334155;border-radius:6px;padding:2px 8px;font-family:monospace;font-size:.85em;color:#7dd3fc}
.step-badge{width:32px;height:32px;border-radius:50%;background:var(--accent);display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0}
footer{background:#0f172a;border-top:1px solid #334155;padding:1.5rem;color:#64748b}
</style>
</head>
<body>
<nav class="navbar navbar-expand-lg">
  <div class="container">
    <a class="navbar-brand" href="/"><i class="fa-solid fa-brain me-2"></i>LLM From Scratch</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#nav">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="nav">
      <ul class="navbar-nav ms-auto gap-1">
        <li class="nav-item"><a class="nav-link {% if request.path=='/' %}active{% endif %}" href="/">Intro</a></li>
        <li class="nav-item"><a class="nav-link {% if '/tokenize' in request.path %}active{% endif %}" href="/tokenize">Tokenization</a></li>
        <li class="nav-item"><a class="nav-link {% if '/architecture' in request.path %}active{% endif %}" href="/architecture">Architecture</a></li>
        <li class="nav-item"><a class="nav-link {% if '/train' in request.path %}active{% endif %}" href="/train">Training</a></li>
        <li class="nav-item"><a class="nav-link {% if '/attention' in request.path %}active{% endif %}" href="/attention">Attention</a></li>
        <li class="nav-item"><a class="nav-link {% if '/generate' in request.path %}active{% endif %}" href="/generate">Generate</a></li>
      </ul>
    </div>
  </div>
</nav>
{% block content %}{% endblock %}
<footer class="text-center mt-5">
  <small>LLM From Scratch — Educational Demo | Flask + PyTorch</small>
</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/js/bootstrap.bundle.min.js"></script>
{% block scripts %}{% endblock %}
</body>
</html>
EOF

# ── templates/index.html ────────────────────────────────────
cat > templates/index.html << 'EOF'
{% extends "base.html" %}
{% block content %}
<div class="hero">
  <div class="container text-center">
    <h1 class="display-5 fw-bold text-white mb-2">🧠 Train an LLM From Scratch</h1>
    <p class="lead text-secondary mb-4">An interactive walkthrough of every stage in building and training a Transformer language model.</p>
    <div class="d-flex justify-content-center gap-3 flex-wrap">
      <a href="/tokenize" class="btn btn-accent px-4"><i class="fa-solid fa-scissors me-2"></i>Start Tutorial</a>
      <a href="/train"    class="btn btn-outline-light px-4"><i class="fa-solid fa-fire me-2"></i>Jump to Training</a>
    </div>
  </div>
</div>
<div class="container my-5">
  <h4 class="fw-bold mb-4 text-center">The LLM Training Pipeline</h4>
  <div class="row g-3 mb-5">
    {% set steps = [
      ("1","fa-scissors","Tokenization","Raw text split into tokens mapped to integer IDs.","/tokenize"),
      ("2","fa-sitemap","Architecture","Transformer: embeddings, multi-head attention, feed-forward blocks.","/architecture"),
      ("3","fa-fire","Training","Feed batches, compute cross-entropy loss, back-propagate, update weights.","/train"),
      ("4","fa-eye","Attention","Visualise learned attention patterns.","/attention"),
      ("5","fa-wand-magic-sparkles","Generation","Sample tokens autoregressively with temperature and top-k.","/generate")
    ] %}
    {% for num,icon,title,desc,link in steps %}
    <div class="col-md-6 col-lg-4">
      <a href="{{ link }}" class="text-decoration-none">
        <div class="card h-100 p-3" style="transition:.2s" onmouseover="this.style.borderColor='#6366f1'" onmouseout="this.style.borderColor='#334155'">
          <div class="d-flex align-items-start gap-3">
            <div class="step-badge">{{num}}</div>
            <div><h6 class="fw-bold mb-1 text-white"><i class="fa-solid {{icon}} me-2"></i>{{title}}</h6>
            <small class="text-secondary">{{desc}}</small></div>
          </div>
        </div>
      </a>
    </div>
    {% endfor %}
  </div>
  <div class="card p-4">
    <h5 class="fw-bold mb-2">What is an LLM?</h5>
    <p class="text-secondary mb-0">A <strong class="text-white">Large Language Model</strong> is a neural network trained to predict the next token in a sequence. In this demo we train <strong class="text-white">TinyGPT</strong> — a decoder-only Transformer — on Shakespeare text. The architecture is identical to GPT-2; only the scale differs.</p>
  </div>
</div>
{% endblock %}
EOF

# ── templates/tokenize.html ─────────────────────────────────
cat > templates/tokenize.html << 'EOF'
{% extends "base.html" %}
{% block content %}
<div class="hero py-3"><div class="container">
  <h2 class="fw-bold mb-1"><i class="fa-solid fa-scissors me-2"></i>Step 1 — Tokenization</h2>
  <p class="text-secondary mb-0">How raw text becomes numbers a neural network can process.</p>
</div></div>
<div class="container my-4">
  <div class="row g-4">
    <div class="col-lg-6">
      <div class="card p-3 mb-3">
        <h6 class="fw-bold mb-2">Enter any text</h6>
        <textarea id="tokInput" class="form-control bg-dark text-light border-secondary" rows="5">To be, or not to be, that is the question.</textarea>
        <button class="btn btn-accent mt-2" onclick="tokenize()"><i class="fa-solid fa-play me-2"></i>Tokenize</button>
      </div>
      <div class="card p-3" id="statsCard" style="display:none">
        <div class="row text-center g-2">
          <div class="col-4"><div class="card p-2"><div id="statTokens" class="fs-4 fw-bold text-info">-</div><small>Tokens</small></div></div>
          <div class="col-4"><div class="card p-2"><div id="statVocab"  class="fs-4 fw-bold text-success">-</div><small>Vocab</small></div></div>
          <div class="col-4"><div class="card p-2"><div id="statRatio"  class="fs-4 fw-bold text-warning">-</div><small>Chars/Tok</small></div></div>
        </div>
      </div>
    </div>
    <div class="col-lg-6">
      <div class="card p-3 mb-3">
        <h6 class="fw-bold mb-2">Token visualisation</h6>
        <div id="tokenViz" class="d-flex flex-wrap gap-1" style="min-height:80px;font-family:monospace">
          <span class="text-secondary"><em>Run tokenization to see results</em></span>
        </div>
      </div>
      <div class="card p-3">
        <h6 class="fw-bold mb-2">Vocabulary</h6>
        <div id="vocabDisplay" class="d-flex flex-wrap gap-1"><span class="text-secondary"><em>-</em></span></div>
      </div>
    </div>
  </div>
  <div class="card p-4 mt-4">
    <h5 class="fw-bold mb-3">Byte-Pair Encoding (BPE)</h5>
    <div id="bpeSteps" class="row g-3"></div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
const C=["#6366f1","#3b82f6","#10b981","#f59e0b","#ec4899","#14b8a6","#8b5cf6","#f97316"];
async function tokenize(){
  const text=document.getElementById("tokInput").value;
  const d=await(await fetch("/api/tokenize",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})})).json();
  document.getElementById("statTokens").textContent=d.num_tokens;
  document.getElementById("statVocab").textContent=d.vocab_size;
  document.getElementById("statRatio").textContent=(d.num_tokens/(text.length||1)).toFixed(2);
  document.getElementById("statsCard").style.display="";
  document.getElementById("tokenViz").innerHTML=d.pairs.map(p=>{
    const col=C[p.id%C.length],disp=p.char===" "?"·":p.char==="\n"?"↵":p.char;
    return `<span title="ID:${p.id}" style="background:${col}22;border:1px solid ${col};border-radius:4px;padding:2px 5px;color:${col}">${disp}<sub style="font-size:.6em;opacity:.7">${p.id}</sub></span>`;
  }).join("");
  document.getElementById("vocabDisplay").innerHTML=d.vocab.map(c=>`<span class="code-pill">${c===" "?"·":c==="\n"?"↵":c}</span>`).join(" ");
  document.getElementById("bpeSteps").innerHTML=d.bpe_steps.map(s=>
    `<div class="col-md-4"><div class="card p-3"><div class="d-flex align-items-center gap-2 mb-2"><div class="step-badge">${s.step}</div><strong>${s.title}</strong></div><small class="text-secondary">${s.desc}</small></div></div>`
  ).join("");
}
tokenize();
</script>
{% endblock %}
EOF

# ── templates/architecture.html ─────────────────────────────
cat > templates/architecture.html << 'EOF'
{% extends "base.html" %}
{% block content %}
<div class="hero py-3"><div class="container">
  <h2 class="fw-bold mb-1"><i class="fa-solid fa-sitemap me-2"></i>Step 2 — Architecture</h2>
  <p class="text-secondary mb-0">Configure the model and see parameter counts update live.</p>
</div></div>
<div class="container my-4">
  <div class="row g-4">
    <div class="col-lg-5">
      <div class="card p-4">
        <h6 class="fw-bold mb-3">Model Configuration</h6>
        <label class="form-label">Embedding dim <span id="lD" class="code-pill">128</span></label>
        <input type="range" class="form-range mb-3" id="dModel" min="32" max="512" step="32" value="128" oninput="update()">
        <label class="form-label">Attention heads <span id="lH" class="code-pill">4</span></label>
        <input type="range" class="form-range mb-1" id="nHeads" min="1" max="16" step="1" value="4" oninput="update()">
        <div id="headWarn" class="text-warning small mb-3" style="display:none">⚠️ d_model must be divisible by n_heads</div>
        <label class="form-label">Layers <span id="lL" class="code-pill">4</span></label>
        <input type="range" class="form-range mb-3" id="nLayers" min="1" max="12" step="1" value="4" oninput="update()">
        <label class="form-label">Context length <span id="lM" class="code-pill">256</span></label>
        <input type="range" class="form-range mb-3" id="maxLen" min="32" max="512" step="32" value="256" oninput="update()">
        <label class="form-label">Vocab size <span id="lV" class="code-pill">65</span></label>
        <input type="range" class="form-range mb-3" id="vocabSize" min="50" max="50257" step="50" value="65" oninput="update()">
      </div>
    </div>
    <div class="col-lg-7">
      <div class="card p-4 mb-3">
        <h6 class="fw-bold mb-3">Parameter Budget</h6>
        <div class="row g-3 text-center">
          <div class="col-6"><div class="card p-3"><div id="totalP" class="fs-3 fw-bold text-info">-</div><small>Total params</small></div></div>
          <div class="col-6"><div class="card p-3"><div id="sizeM"  class="fs-3 fw-bold text-success">-</div><small>Size (MB fp32)</small></div></div>
          <div class="col-4"><div class="card p-2"><div id="embP"   class="fw-bold text-warning">-</div><small class="text-secondary">Embedding</small></div></div>
          <div class="col-4"><div class="card p-2"><div id="attnP"  class="fw-bold" style="color:#ec4899">-</div><small class="text-secondary">Attn/layer</small></div></div>
          <div class="col-4"><div class="card p-2"><div id="ffP"    class="fw-bold" style="color:#14b8a6">-</div><small class="text-secondary">FF/layer</small></div></div>
        </div>
      </div>
      <div class="card p-4">
        <h6 class="fw-bold mb-3">Architecture Diagram</h6>
        <pre style="background:#0f172a;padding:1rem;border-radius:8px;color:#7dd3fc;font-size:.78rem">
Input [B, T]
  Token Emb + Pos Emb -> [B, T, D]
  [ TransformerBlock x <span id="diagL">4</span> ]
    LayerNorm -> Multi-Head Attention (causal)
    LayerNorm -> FeedForward GELU (D->4D->D)
  LayerNorm -> Linear -> Vocab
Output logits [B, T, Vocab]</pre>
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
function fmt(n){if(n>=1e9)return(n/1e9).toFixed(2)+"B";if(n>=1e6)return(n/1e6).toFixed(2)+"M";if(n>=1e3)return(n/1e3).toFixed(0)+"K";return n;}
async function update(){
  const d=+document.getElementById("dModel").value,h=+document.getElementById("nHeads").value,
        l=+document.getElementById("nLayers").value,m=+document.getElementById("maxLen").value,
        v=+document.getElementById("vocabSize").value;
  document.getElementById("lD").textContent=d;document.getElementById("lH").textContent=h;
  document.getElementById("lL").textContent=l;document.getElementById("lM").textContent=m;
  document.getElementById("lV").textContent=v;document.getElementById("diagL").textContent=l;
  document.getElementById("headWarn").style.display=(d%h!==0)?"":"none";
  const s=await(await fetch("/api/model_stats",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({d_model:d,n_heads:h,n_layers:l,max_len:m,vocab_size:v})})).json();
  document.getElementById("totalP").textContent=fmt(s.total_params);
  document.getElementById("sizeM").textContent=s.size_mb+" MB";
  document.getElementById("embP").textContent=fmt(s.embedding_params);
  document.getElementById("attnP").textContent=fmt(s.attn_params_per_layer);
  document.getElementById("ffP").textContent=fmt(s.ff_params_per_layer);
}
update();
</script>
{% endblock %}
EOF

# ── templates/train.html ────────────────────────────────────
cat > templates/train.html << 'EOF'
{% extends "base.html" %}
{% block content %}
<div class="hero py-3"><div class="container">
  <h2 class="fw-bold mb-1"><i class="fa-solid fa-fire me-2"></i>Step 3 — Training</h2>
  <p class="text-secondary mb-0">Train TinyGPT on Shakespeare and watch the loss drop live.</p>
</div></div>
<div class="container my-4">
  <div class="row g-4">
    <div class="col-lg-4">
      <div class="card p-4">
        <h6 class="fw-bold mb-3">Training Config</h6>
        <label class="form-label">Steps <span id="lS" class="code-pill">500</span></label>
        <input type="range" class="form-range mb-3" id="cfgSteps" min="100" max="2000" step="100" value="500" oninput="document.getElementById('lS').textContent=this.value">
        <label class="form-label">Learning rate</label>
        <select class="form-select bg-dark text-light border-secondary mb-3" id="cfgLR">
          <option value="1e-3">1e-3</option><option value="3e-4" selected>3e-4</option><option value="1e-4">1e-4</option>
        </select>
        <label class="form-label">Batch size <span id="lB" class="code-pill">32</span></label>
        <input type="range" class="form-range mb-3" id="cfgBatch" min="8" max="128" step="8" value="32" oninput="document.getElementById('lB').textContent=this.value">
        <label class="form-label">d_model <span id="lDM" class="code-pill">64</span></label>
        <input type="range" class="form-range mb-3" id="cfgDM" min="32" max="256" step="32" value="64" oninput="document.getElementById('lDM').textContent=this.value">
        <label class="form-label">Layers <span id="lLy" class="code-pill">2</span></label>
        <input type="range" class="form-range mb-4" id="cfgLy" min="1" max="6" step="1" value="2" oninput="document.getElementById('lLy').textContent=this.value">
        <div class="d-flex gap-2">
          <button id="btnStart" class="btn btn-accent flex-fill" onclick="startTraining()"><i class="fa-solid fa-play me-2"></i>Start</button>
          <button id="btnStop"  class="btn btn-outline-danger flex-fill" onclick="stopTraining()" disabled><i class="fa-solid fa-stop me-2"></i>Stop</button>
        </div>
        <div id="trainStatus" class="text-center mt-3 text-secondary small">Idle</div>
      </div>
    </div>
    <div class="col-lg-8">
      <div class="card p-4 mb-3">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <h6 class="fw-bold mb-0">Training Loss (live)</h6>
          <span id="lossLabel" class="badge bg-secondary">-</span>
        </div>
        <canvas id="lossChart" height="200"></canvas>
      </div>
      <div class="card p-4">
        <h6 class="fw-bold mb-1">Cross-entropy loss</h6>
        <p class="text-secondary small mb-1">Loss = -log P(correct token). Random baseline on 65 chars ≈ 4.17. Target after training: &lt; 1.5.</p>
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
let chart,timer,last=0;
chart=new Chart(document.getElementById("lossChart"),{type:"line",data:{labels:[],datasets:[{label:"Loss",data:[],borderColor:"#6366f1",backgroundColor:"rgba(99,102,241,.15)",fill:true,tension:.3,pointRadius:0}]},options:{responsive:true,animation:false,scales:{x:{ticks:{color:"#64748b",maxTicksLimit:8}},y:{ticks:{color:"#64748b"}}},plugins:{legend:{display:false}}}});
async function startTraining(){
  const cfg={steps:+document.getElementById("cfgSteps").value,lr:+document.getElementById("cfgLR").value,
    batch_size:+document.getElementById("cfgBatch").value,d_model:+document.getElementById("cfgDM").value,
    n_layers:+document.getElementById("cfgLy").value,block_size:64,n_heads:4};
  await fetch("/api/stop_training",{method:"POST"});await new Promise(r=>setTimeout(r,300));
  chart.data.labels=[];chart.data.datasets[0].data=[];chart.update();last=0;
  const d=await(await fetch("/api/start_training",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(cfg)})).json();
  if(d.error){alert(d.error);return;}
  document.getElementById("btnStart").disabled=true;document.getElementById("btnStop").disabled=false;
  timer=setInterval(poll,400);
}
async function stopTraining(){
  await fetch("/api/stop_training",{method:"POST"});clearInterval(timer);
  document.getElementById("btnStart").disabled=false;document.getElementById("btnStop").disabled=true;
}
async function poll(){
  const d=await(await fetch("/api/training_status")).json();
  if(d.losses.length>last){
    d.losses.slice(last).forEach((l,i)=>{chart.data.labels.push(last+i+1);chart.data.datasets[0].data.push(l);});
    last=d.losses.length;chart.update("none");
    const cur=d.losses[d.losses.length-1];
    document.getElementById("lossLabel").textContent="Loss: "+cur.toFixed(4);
    document.getElementById("lossLabel").className="badge "+(cur<2?"bg-success":cur<3?"bg-warning":"bg-danger");
  }
  const pct=d.total_steps>0?Math.round(100*d.step/d.total_steps):0;
  document.getElementById("trainStatus").textContent=`Step ${d.step}/${d.total_steps} (${pct}%) — ${d.status}`;
  if(!d.is_running&&d.status!=="idle"){clearInterval(timer);document.getElementById("btnStart").disabled=false;document.getElementById("btnStop").disabled=true;}
}
</script>
{% endblock %}
EOF

# ── templates/attention.html ────────────────────────────────
cat > templates/attention.html << 'EOF'
{% extends "base.html" %}
{% block content %}
<div class="hero py-3"><div class="container">
  <h2 class="fw-bold mb-1"><i class="fa-solid fa-eye me-2"></i>Step 4 — Attention Visualisation</h2>
  <p class="text-secondary mb-0">See what each token attends to. Requires a trained model.</p>
</div></div>
<div class="container my-4">
  <div class="row g-4">
    <div class="col-lg-5">
      <div class="card p-4">
        <h6 class="fw-bold mb-3">Inspect Attention</h6>
        <input id="attnText" class="form-control bg-dark text-light border-secondary mb-3" value="To be, or not" maxlength="32" placeholder="Input text (max 32 chars)"/>
        <label class="form-label">Layer <span id="lLayer" class="code-pill">0</span></label>
        <input type="range" class="form-range mb-3" id="attnLayer" min="0" max="5" value="0" oninput="document.getElementById('lLayer').textContent=this.value">
        <button class="btn btn-accent w-100" onclick="getAttn()"><i class="fa-solid fa-magnifying-glass me-2"></i>Visualise</button>
        <div id="attnErr" class="text-warning mt-2 small"></div>
      </div>
      <div class="card p-4 mt-3">
        <h6 class="fw-bold mb-2">Reading the heatmap</h6>
        <small class="text-secondary">Row = query token. Column = key attended to. Darker = higher weight. Upper triangle = zero (causal mask).</small>
      </div>
    </div>
    <div class="col-lg-7">
      <div class="card p-4" style="min-height:280px">
        <h6 class="fw-bold mb-3">Attention heads <span id="headLabel" class="text-secondary small">(visualise first)</span></h6>
        <div id="headGrid" class="row g-3"></div>
      </div>
    </div>
  </div>
  <div class="card p-4 mt-4">
    <h5 class="fw-bold mb-2">Scaled Dot-Product Attention</h5>
    <code class="d-block p-3 rounded text-info mb-2" style="background:#0f172a">Attention(Q,K,V) = softmax( QK^T / sqrt(d_k) ) * V</code>
    <small class="text-secondary">Q = Query (what am I looking for?), K = Key (what do I offer?), V = Value (what info do I carry?)</small>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
async function getAttn(){
  document.getElementById("attnErr").textContent="";
  const text=document.getElementById("attnText").value,layer=+document.getElementById("attnLayer").value;
  const r=await fetch("/api/attention",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,layer})});
  const d=await r.json();if(d.error){document.getElementById("attnErr").textContent=d.error;return;}
  document.getElementById("headLabel").textContent=`Layer ${layer}, ${d.heads.length} heads`;
  const sz=d.tokens.length,cs=Math.min(Math.floor(200/sz),28);
  document.getElementById("headGrid").innerHTML=d.heads.map((_,hi)=>
    `<div class="col-6 col-lg-4"><div class="card p-2 text-center"><small class="text-secondary mb-1 d-block">Head ${hi}</small>
    <canvas id="h${hi}" width="${sz*cs}" height="${sz*cs}" style="image-rendering:pixelated;max-width:100%"></canvas></div></div>`
  ).join("");
  d.heads.forEach((head,hi)=>{
    const cv=document.getElementById(`h${hi}`),c=cv.getContext("2d"),w=cv.width/sz,h=cv.height/sz;
    head.forEach((row,r)=>row.forEach((val,col)=>{
      c.fillStyle=col>r?"rgba(0,0,0,0)":`rgba(99,102,241,${Math.pow(val,.5)})`;c.fillRect(col*w,r*h,w,h);
    }));
    c.fillStyle="#94a3b8";c.font=`${Math.min(w*.7,12)}px monospace`;
    d.tokens.forEach((t,i)=>c.fillText(t===" "?"·":t==="\n"?"↵":t,i*w+2,cv.height-2));
  });
}
</script>
{% endblock %}
EOF

# ── templates/generate.html ─────────────────────────────────
cat > templates/generate.html << 'EOF'
{% extends "base.html" %}
{% block content %}
<div class="hero py-3"><div class="container">
  <h2 class="fw-bold mb-1"><i class="fa-solid fa-wand-magic-sparkles me-2"></i>Step 5 — Text Generation</h2>
  <p class="text-secondary mb-0">Generate text using your trained model. Train first!</p>
</div></div>
<div class="container my-4">
  <div class="row g-4">
    <div class="col-lg-4">
      <div class="card p-4">
        <h6 class="fw-bold mb-3">Generation Settings</h6>
        <label class="form-label">Prompt</label>
        <input id="genPrompt" class="form-control bg-dark text-light border-secondary mb-3" value="First Citizen:" maxlength="64"/>
        <label class="form-label">Tokens <span id="lTok" class="code-pill">200</span></label>
        <input type="range" class="form-range mb-3" id="cfgTok" min="50" max="500" step="50" value="200" oninput="document.getElementById('lTok').textContent=this.value">
        <label class="form-label">Temperature <span id="lTemp" class="code-pill">1.0</span></label>
        <input type="range" class="form-range mb-1" id="cfgTemp" min="0.1" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('lTemp').textContent=(+this.value).toFixed(1)">
        <small class="text-secondary d-block mb-3">Low = predictable | High = creative</small>
        <label class="form-label">Top-k <span id="lK" class="code-pill">off</span></label>
        <input type="range" class="form-range mb-4" id="cfgK" min="0" max="40" step="1" value="0" oninput="document.getElementById('lK').textContent=this.value==0?'off':this.value">
        <button class="btn btn-accent w-100" onclick="generate()"><i class="fa-solid fa-wand-magic-sparkles me-2"></i>Generate</button>
        <div id="genErr" class="text-warning mt-2 small"></div>
      </div>
    </div>
    <div class="col-lg-8">
      <div class="card p-4 mb-3">
        <div class="d-flex justify-content-between mb-2">
          <h6 class="fw-bold mb-0">Output</h6>
          <button class="btn btn-sm btn-outline-secondary" onclick="navigator.clipboard.writeText(document.getElementById('out').textContent)">Copy</button>
        </div>
        <div id="out" style="background:#0f172a;border-radius:8px;padding:1rem;font-family:monospace;min-height:180px;white-space:pre-wrap;font-size:.9rem">
          <span class="text-secondary">Output will appear here after generation...</span>
        </div>
      </div>
      <div class="card p-4">
        <h5 class="fw-bold mb-3">Decoding Strategies</h5>
        <table class="table table-dark table-bordered mb-0">
          <thead><tr><th>Strategy</th><th>How</th><th>Best for</th></tr></thead>
          <tbody>
            <tr><td>Greedy</td><td>Always argmax</td><td>Short factual answers</td></tr>
            <tr><td>Temperature</td><td>Sample tempered dist.</td><td>Creative text</td></tr>
            <tr><td>Top-k</td><td>Sample from top k only</td><td>General purpose</td></tr>
            <tr><td>Top-p</td><td>Smallest set summing to p</td><td>State-of-the-art chat</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
async function generate(){
  document.getElementById("genErr").textContent="";
  document.getElementById("out").textContent="Generating...";
  const r=await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
    prompt:document.getElementById("genPrompt").value,
    max_tokens:+document.getElementById("cfgTok").value,
    temperature:+document.getElementById("cfgTemp").value,
    top_k:+document.getElementById("cfgK").value||null
  })});
  const d=await r.json();
  if(d.error){document.getElementById("out").innerHTML=`<span class="text-warning">${d.error}</span>`;return;}
  document.getElementById("out").innerHTML=`<span style="color:#6366f1;font-weight:700">${d.prompt}</span>${d.generated}`;
}
</script>
{% endblock %}
EOF

echo ""
echo "✅ Done! All files created in ./$PROJECT/"
echo ""
echo "Next steps:"
echo "  cd $PROJECT"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
echo "  pip install -r requirements.txt"
echo "  python app.py"
echo ""
echo "Then open http://localhost:7860 🎉"