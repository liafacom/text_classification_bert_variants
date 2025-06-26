# !pip install accelerate rich tqdm # para o transformers
# !pip install flash-attn --no-build-isolation # para o modernbert
# !wandb init

import os
import torch
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, get_scheduler
from torch import nn
from torch.optim import AdamW
from text_dataset.reader import (
    get_r8,
    get_ohsumed,
    get_r52,
    get_20newsgroups,
    get_mpqa,
    best_max_lenght,
    get_mr,
    get_trec_6,
    get_agnew,
    get_snippets,
    get_sst2,
    get_overruling,
)

MODEL_NAMES = [
    "bert-base-uncased",
    "distilbert/distilbert-base-uncased",
    "FacebookAI/roberta-base",
    "microsoft/deberta-v3-base",
    "microsoft/mpnet-base",
    "answerdotai/ModernBERT-base",
]

EMBEDDING_TYPES = [
    "CLS_Frozen",
    "CLS_Finetune",
    "AVG_Frozen",
    "AVG_Finetune",
    "AVG_wo_CLS_Frozen",
    "AVG_wo_CLS_Finetune",
    "FirstLastAVG_Frozen",
    "FirstLastAVG_Finetune",
    "FirstLastAVG_wo_CLS_Frozen",
    "FirstLastAVG_wo_CLS_Finetune",
]


def load_data(filepath):
    df = pd.read_csv(filepath) 
    return df


def split_data(train_df, test_df):
    train_df_, val_df = train_test_split(
        train_df, test_size=0.1, random_state=SEED, stratify=train_df.label
    )
    return train_df_, val_df, test_df


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.inputs = {
            key: val
            for key, val in self.encodings.items()
            if key in ["input_ids", "attention_mask", "token_type_ids"]
        }
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TransformerClassifier(nn.Module):
    def __init__(self, model_name, embedding_type, num_labels):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.embedding_type = embedding_type
        self.num_labels = num_labels
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        if "Frozen" in embedding_type:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, **inputs):
        labels = inputs.pop("labels", None)
        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states  # Pegando todas as camadas do modelo

        if "CLS" in self.embedding_type:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        elif "FirstLastAVG" in self.embedding_type:
            first_layer = hidden_states[1]  # Primeira camada oculta
            last_layer = hidden_states[-1]  # Última camada oculta
            token_embeddings = (first_layer + last_layer) / 2
            if "wo_CLS" in self.embedding_type:
                token_embeddings = token_embeddings[:, 1:, :]
            pooled_output = token_embeddings.mean(dim=1)
        else:
            token_embeddings = outputs.last_hidden_state
            if "wo_CLS" in self.embedding_type:
                token_embeddings = token_embeddings[:, 1:, :]
            pooled_output = token_embeddings.mean(dim=1)

        logits = self.classifier(pooled_output)

        # Se labels forem fornecidos, calcula a loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else logits


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(pred):
    global model
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)

    # Gerar matriz de confusão
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(labels),
        yticklabels=np.unique(labels),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Log no wandb
    wandb.log(
        {
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": wandb.Image(f"./{directory}/confusion_matrix.png"),
        }
    )
    for name, param in model.named_parameters():
        if "weight" in name:
            wandb.log(
                {f"histogram/{name}": wandb.Histogram(param.detach().cpu().numpy())}
            )

    layer_magnitudes = {
        name: torch.norm(param).item()
        for name, param in model.named_parameters()
        if "weight" in name
    }
    wandb.log(
        {
            "weight_magnitudes": wandb.Table(
                columns=["Layer", "Magnitude"], data=list(layer_magnitudes.items())
            )
        }
    )

    # Print classification report
    print(
        "Classification Report:\n",
        classification_report(labels, preds, zero_division=0, digits=4),
    )
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def evaluate(model, dataloader, prefix="Val"):
    model.eval()
    predictions, true_labels = [], []

    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())
            loss = outputs["loss"]

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    precision = precision_score(
        true_labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    # Gerar matriz de confusão
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(true_labels),
        yticklabels=np.unique(true_labels),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"./{directory}/confusion_matrix.png")
    plt.close()

    test_metrics = {
        f"{prefix}_loss": avg_loss,
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_f1": f1,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_confusion_matrix": wandb.Image(f"./{directory}/confusion_matrix.png"),
    }
    # Print classification report
    print(
        "Classification Report:\n",
        classification_report(true_labels, predictions, zero_division=0, digits=4),
    )
    print(f"{prefix} Accuracy: {accuracy:.4f}, {prefix} F1: {f1:.4f}")
    wandb.log(test_metrics)

    return test_metrics


def clean_up(folders = ["./results", "./logs"]):
    """Remove arquivos temporários para evitar acúmulo de dados."""
    
    for folder in folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Erro ao remover {file_path}: {e}")


def train_and_evaluate(
    model_name,
    embedding_type,
    train_df,
    val_df,
    test_df,
    dataset_name,
    batch_size=16,
    max_epochs=10,
    patience=2,
    lr=2e-5,
    weight_decay=0.01,
    tags=[],
    metric_eval="val_accuracy",
):
    global model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = best_max_lenght()[dname]

    train_dataset = TextDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    val_dataset = TextDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    test_dataset = TextDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )

    model = TransformerClassifier(
        model_name, embedding_type, num_labels=len(set(train_df["label"]))
    ).to(dev)

    wrun = wandb.init(project="transf-benchmark-clf-classic", tags=tags)
    wrun.config.name_config = f"{model_name}_{embedding_type}"
    wrun.config.transformer_model_name = model_name
    wrun.config.dataset_name = dataset_name
    wrun.config.embedding_type = embedding_type
    wrun.config.patience = patience
    wrun.config.weight_decay = weight_decay
    wrun.config.lr = lr
    wrun.config.seed = SEED

    model = TransformerClassifier(
        model_name, embedding_type, num_labels=len(set(train_df["label"]))
    ).to(dev)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = len(train_loader) * max_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% do treinamento como warmup

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_metric = 0
    epochs_no_improve = 0
    best_epoch = 0
    single_model_name = model_name.split("/")[-1]
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        wandb.log({"loss_train": avg_loss})

        # Avaliação no conjunto de validação
        val_metrics = evaluate(model, val_loader, prefix="val")
        current_metric = val_metrics[metric_eval]
        wandb.log(
            {
                "val_f1": val_metrics["val_f1"],
                "val_accuracy": val_metrics["val_accuracy"],
                "val_precision": val_metrics["val_precision"],
                "val_recall": val_metrics["val_recall"],
            }
        )

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, {metric_eval}={current_metric:.4f}")

        if current_metric > best_metric:
            print(f"Improving from {best_metric:.5f} to {current_metric:.5f}, epoch: {epoch + 1}.")
            best_metric = current_metric
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                model.state_dict(),
                f"./{directory}/best_model_{single_model_name}_{embedding_type}.pt",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping ativado após {epoch + 1} épocas.")
                break

    # Avaliação final no conjunto de teste
    model.load_state_dict(
        torch.load(
            f"./{directory}/best_model_{single_model_name}_{embedding_type}.pt",
            weights_only=True,
        )
    )
    test_metrics = evaluate(model, test_loader, prefix="test")
    wrun.config.best_epoch = best_epoch
    wrun.config.best_metric = best_metric
    wrun.config.metric_eval = metric_eval
    del test_metrics["test_confusion_matrix"]
    wrun.config.update(test_metrics)
    

    wandb.finish()

    clean_up(folders=[directory])


model = None

import argparse


def get_dataset(dataset_name):
    func = None
    if dataset_name == "OH":
        func = get_ohsumed
    elif dataset_name == "MPQA":
        func = get_mpqa
    elif dataset_name == "R8":
        func = get_r8
    elif dataset_name == "R52":
        func = get_r52
    elif dataset_name == "MR":
        func = get_mr
    elif dataset_name == "AG":
        func = get_agnew
    elif dataset_name == "SN":
        func = get_snippets
    elif dataset_name == "SST2":
        func = get_sst2
    elif dataset_name == "TREC":
        func = get_trec_6
    elif dataset_name == "20NG":
        func = get_20newsgroups
    elif dataset_name == "OVER":
        func = get_overruling
    else:
        func = get_mr
    dtr, dte, labels_name, dname = func()
    train_df, val_df, test_df = split_data(dtr, dte)
    return train_df, val_df, test_df, dname


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

# python benchmark_transformers.py --seed 0 --dataset R52 --epochs 10 --batch_size 16
# python3 benchmark_transformers.py --seed 0 --dataset R8 --epochs 10 --batch_size 16
directory = None

if __name__ == "__main__":
    # Criando o parser
    parser = argparse.ArgumentParser(description="Seed")

    # Adicionando o argumento seed (inteiro)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Número inteiro para definir a seed do experimento.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Nome do dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Número inteiro para definir a qtd epocas do experimento.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Número inteiro para definir cuda gpu.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Número inteiro para definir o batch_size do experimento.",
    )
    # Adicionando o argumento que recebe uma lista
    parser.add_argument(
        "--tags",
        nargs="+",  # Permite múltiplos valores
        type=str,
        default=["test"],  # Valor padrão caso não seja fornecido
        help="Lista de strings separadas por espaço. Padrão: ['test']",
    )
    
    # Adicionando o argumento que recebe uma lista
    parser.add_argument(
        "--emb",
        nargs="+",  # Permite múltiplos valores
        type=str,
        default=None,  # Valor padrão caso não seja fornecido
        help="Lista de strings separadas por espaço. Padrão: None",
    )
    
    # Adicionando o argumento que recebe uma lista
    parser.add_argument(
        "--models",
        nargs="+",  # Permite múltiplos valores
        type=str,
        default=None,  # Valor padrão caso não seja fornecido
        help="Lista de strings separadas por espaço. Padrão: None",
    )

    # Parseando os argumentos
    args = parser.parse_args()

    # Exibindo o valor da seed
    print(f"Seed escolhida: {args.seed}")

    # Configurações globais
    SEED = args.seed
    epochs = args.epochs
    batch_size = args.batch_size
    dataset_name = args.dataset
    embs = args.emb
    models = args.models
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    tags = args.tags

    directory = f"logs_{dataset_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # torch.manual_seed(SEED)
    np.random.seed(SEED)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, dname = get_dataset(dataset_name)
    embeddings_type = EMBEDDING_TYPES
    if embs is not None:
        embeddings_type = embs
        
    models_name = MODEL_NAMES
    if models is not None:
        models_name = models

    for model_name in models_name:
        for embedding_type in embeddings_type:
            train_and_evaluate(
                model_name,
                embedding_type,
                train_df,
                val_df,
                test_df,
                dname,
                batch_size=batch_size,
                max_epochs=epochs,
                patience=10,
                tags=tags,
            )
