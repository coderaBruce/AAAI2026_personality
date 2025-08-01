# embedding 模型from Hailong, chunk
# regression 好像不存在类别不均匀问题
import os, sys
import argparse

import torch
import time
import json
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
from torch.utils.data import WeightedRandomSampler

from transformers import RobertaTokenizer, RobertaModel
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr




sys.stdout.reconfigure(line_buffering=True)
os.system("nvidia-smi")



# Fix seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Config
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
parser.add_argument("--disease", type=str, default="GT_Extraversion", choices=["GT_Extraversion", "GT_Agreeableness", "GT_Conscientiousness", "GT_Neuroticism", "GT_Openness"], help="Disease type")

args = parser.parse_args()
LORA_RANK = args.lora_rank
DISEASE = args.disease


EPOCHS = 100
BATCH_SIZE = 8
USE_LORA_OPTIONS = [True] #  [True, False]
POOLING_TYPE = "mean"
OVERLAP_RATIO = 0.5
CHUNK_SIZE = 128
# LORA_RANK = 32
# DISEASE = "GT_Extraversion"   # "GT_Agreeableness", "GT_Conscientiousness", "GT_Neuroticism", "GT_Openness"
TRAIN_PATH = f"../personalization_data/big_five/encoder_{DISEASE}_train.jsonl"
VAL_PATH = f"../personalization_data/big_five/encoder_{DISEASE}_test.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"========= Using device: {DEVICE} ==========")
print(f"========= Using lora rank: {LORA_RANK} ==========")
print(f"========= Using disease: {DISEASE} ==========")

# Load data
def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            data.append([entry["text"], entry["label"]])
    return pd.DataFrame(data, columns=["text", "label"])

train_data = load_data(TRAIN_PATH)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Estimate max_chunks
def estimate_max_chunks(texts, chunk_size=512, overlap_ratio=0.0):
    stride = int(chunk_size * (1 - overlap_ratio))
    chunk_counts = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        total_len = len(tokens)
        if total_len < chunk_size:
            chunk_counts.append(1)
        else:
            num_chunks = max(1, (total_len - chunk_size) // stride + 1)
            chunk_counts.append(num_chunks)
    return int(np.percentile(chunk_counts, 95))

MAX_CHUNKS = estimate_max_chunks(train_data["text"], CHUNK_SIZE, OVERLAP_RATIO)

# Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, chunk_size=512, overlap_ratio=0.0, max_chunks=8):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.max_chunks = max_chunks

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").squeeze()
        attention_mask = torch.ones_like(tokens)
        stride = int(self.chunk_size * (1 - self.overlap_ratio))
        total_len = tokens.size(0)
        if total_len < self.chunk_size:
            pad_len = self.chunk_size - total_len
            tokens = F.pad(tokens, (0, pad_len), value=self.tokenizer.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        input_ids_chunks = tokens.unfold(0, self.chunk_size, stride)
        attention_mask_chunks = attention_mask.unfold(0, self.chunk_size, stride)
        num_chunks = input_ids_chunks.size(0)
        if num_chunks < self.max_chunks:
            pad_len = self.max_chunks - num_chunks
            pad_tensor = torch.full((pad_len, self.chunk_size), self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids_chunks = torch.cat([input_ids_chunks, pad_tensor], dim=0)
            attention_mask_chunks = torch.cat([attention_mask_chunks, torch.zeros((pad_len, self.chunk_size), dtype=torch.long)], dim=0)
        else:
            input_ids_chunks = input_ids_chunks[:self.max_chunks]
            attention_mask_chunks = attention_mask_chunks[:self.max_chunks]
        return {
            'input_ids': input_ids_chunks,
            'attention_mask': attention_mask_chunks,
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Model
class BertRegressor(nn.Module):
    def __init__(self, bert_model="roberta-base", use_lora=False, pooling_type="mean", lstm_hidden_size=512):
        super().__init__()
        base_model = RobertaModel.from_pretrained(bert_model)
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=LORA_RANK,
                lora_alpha=LORA_RANK,
                lora_dropout=0.1,
                target_modules=["query", "value"]
            )
            self.bert = get_peft_model(base_model, lora_config)
        else:
            self.bert = base_model
            # free bert parameters
            for param in self.bert.parameters():
                param.requires_grad = False

        self.pooling_type = pooling_type
        hidden_dim = self.bert.config.hidden_size
        if pooling_type == "lstm":
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_size, batch_first=True)
            final_dim = lstm_hidden_size
        else:
            final_dim = hidden_dim
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(final_dim),
        #     nn.Linear(final_dim, final_dim),
        #     nn.ReLU(),
        #     nn.Linear(final_dim, output_size)
        # )
        self.regressor = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1)  # 输出一个连续值
)

        # self.classifier = nn.Linear(final_dim, output_size)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len, chunk_size = input_ids.size()
        input_ids = input_ids.view(-1, chunk_size).to(DEVICE)
        attention_mask = attention_mask.view(-1, chunk_size).to(DEVICE)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = bert_output.last_hidden_state[:, 0, :].view(batch_size, seq_len, -1)
        if self.pooling_type == "mean":
            pooled = torch.mean(cls_embeddings, dim=1)
        elif self.pooling_type == "max":
            pooled, _ = torch.max(cls_embeddings, dim=1)
        elif self.pooling_type == "lstm":
            _, (hidden, _) = self.lstm(cls_embeddings)
            pooled = hidden.squeeze(0)
        
        logits = self.regressor(pooled)
        return logits  # ← 不再用 sigmoid
    

# evaluate
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.squeeze(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    try:
        pearson_r, _ = pearsonr(all_labels, all_preds)
    except:
        pearson_r = 0.0

    try:
        spearman_r, _ = spearmanr(all_labels, all_preds)
    except:
        spearman_r = 0.0
    return mse, mae, r2, pearson_r, spearman_r

# Run comparison
for use_lora in USE_LORA_OPTIONS:
    tag = "LoRA" if use_lora else "Full"
    print(f"\n==== Training with {tag} Fine-Tuning ====")


    train_dataset = SentimentDataset(train_data["text"].to_numpy(), train_data["label"].to_numpy(), tokenizer,
                                     chunk_size=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO, max_chunks=MAX_CHUNKS)
    
    # 计算类别权重
    # train_labels = train_data["label"].to_numpy()
    # class_counts = np.bincount(train_labels)
    # weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # samples_weights = weights[train_labels]

    # # 创建 sampler
    # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    # # 构建 DataLoader
    # train_dataset = SentimentDataset(train_data["text"].to_numpy(), train_data["label"].to_numpy(), tokenizer,
    #                                 chunk_size=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO, max_chunks=MAX_CHUNKS)
    # train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_data = load_data(VAL_PATH)
    val_dataset = SentimentDataset(val_data["text"].to_numpy(), val_data["label"].to_numpy(), tokenizer,
                                chunk_size=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO, max_chunks=MAX_CHUNKS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BertRegressor(use_lora=use_lora, pooling_type=POOLING_TYPE).to(DEVICE)
    print(" Checking LoRA parameters are trainable...")
    for name, param in model.named_parameters():
        if "lora" in name:
            print(f"{name}: requires_grad = {param.requires_grad}")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    # optimizer = SGD(model.parameters(), lr=5e-4)
    # class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_data["label"]), y=train_data["label"])
    # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE))
    
    # class_weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=np.unique(train_data["label"]),
    #     y=train_data["label"]
    # )
    # class_weights_tensor = torch.tensor(torch.tensor([0.01, 1.0]), dtype=torch.float).to(DEVICE)
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    loss_fn = nn.MSELoss()

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total = sum(p.numel() for p in model.parameters())
    trainable = count_trainable_parameters(model)
    print(f"Total Params: {total:,}, Trainable: {trainable:,} ({trainable/total:.2%})")

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    train_losses = []
    start_time = time.time()
    best_mse, best_mae, best_r2, best_pearson_r, best_spearman_r = float("inf"), float("inf"), -float("inf"), -float("inf"), -float("inf")
    best_metrics = {"epoch": -1, "mse": float("inf"), "mae": float("inf"), "r2": -float("inf"), "pearson_r": -float("inf"), "spearman_r": -float("inf")}

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        model.train()

        for name, param in model.named_parameters():
            if 'classifier' in name:
                print(f"{name}: requires_grad = {param.requires_grad}, shape = {tuple(param.shape)}, num_params = {param.numel()}")

        for i, batch in enumerate(train_loader): 
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(1)
            loss = loss_fn(outputs, labels)

            # L2 regularization
            l2_lambda = 1e-4
            l2_reg = sum(torch.norm(param, p=2) for param in model.regressor.parameters())
            loss += l2_lambda * l2_reg

            print(f"[Epoch {epoch}, Step {i}] Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 20 == 0:
                mse, mae, r2, pearson_r, spearman_r = evaluate(model, val_loader)
                print(f"✅ Eval Epoch {epoch} Batch {i} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Pearson R: {pearson_r:.4f}, Spearman R: {spearman_r:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

                if mse < best_mse:
                    best_metrics.update({
                        "epoch": epoch,
                        "mse": mse,
                        "mae": mae,
                        "r2": r2,
                        "pearson_r": pearson_r,
                        "spearman_r": spearman_r
                    })
                best_mse = min(best_mse, mse)
                best_mae = min(best_mae, mae)
                best_r2 = max(best_r2, r2)
                best_pearson_r = max(best_pearson_r, pearson_r)
                best_spearman_r = max(best_spearman_r, spearman_r)
                model.train()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")

    # 🔚 Final best metrics
    print(f"\n🏅 Best MSE at Epoch {best_metrics['epoch'] + 1}:")
    print(f"where MSE: {best_metrics['mse']:.4f}, MAE: {best_metrics['mae']:.4f}, R2: {best_metrics['r2']:.4f}, Pearson R: {best_metrics['pearson_r']:.4f}, Spearman R: {best_metrics['spearman_r']:.4f}")
    print(f"🔸 MSE: {best_mse:.4f}")
    print(f"🔸 MAE: {best_mae:.4f}")
    print(f"🔸 R2 : {best_r2:.4f}")
    print(f"🔸 Pearson R: {best_pearson_r:.4f}")
    print(f"🔸 Spearman R: {best_spearman_r:.4f}")




    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f}s")
    if DEVICE.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
        print(f"Max GPU Memory Used: {peak_mem:.2f} MB")






