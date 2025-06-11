import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow
import mlflow.pyfunc
import numpy as np

class PTDistilBertClassifier:
    def __init__(self, num_classes: int, model_name: str = "distilbert-base-uncased"):
        self.num_classes = num_classes
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts, labels=None, max_length=None):
        max_length = max_length or self.tokenizer.model_max_length
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        if labels is not None:
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    def fit(
        self, texts, labels,
        epochs=3, lr=3e-5, batch_size=8, val_split=0.8,
        model_save_path: str = None
    ):
        # split
        from sklearn.model_selection import train_test_split
        x_tr, x_val, y_tr, y_val = train_test_split(
            texts, labels, train_size=val_split, random_state=42
        )

        # data loaders
        def make_loader(x, y):
            enc = self.encode(x, y)
            ds = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
            return DataLoader(ds, batch_size=batch_size, shuffle=True)

        train_loader = make_loader(x_tr, y_tr)
        val_loader   = make_loader(x_val, y_val)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # train
            self.model.train()
            total_loss = 0
            for ids, masks, labs in train_loader:
                ids, masks, labs = [t.to(self.device) for t in (ids, masks, labs)]
                out = self.model(input_ids=ids, attention_mask=masks, labels=labs)
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # val
            self.model.eval()
            all_preds, all_probs, all_labels = [], [], []
            with torch.no_grad():
                for ids, masks, labs in val_loader:
                    ids, masks = ids.to(self.device), masks.to(self.device)
                    logits = self.model(input_ids=ids, attention_mask=masks).logits
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    all_probs.append(probs)
                    all_preds.extend(preds)
                    all_labels.extend(labs.numpy())

            # all_probs = sum(all_probs)  
            all_probs = np.vstack(all_probs)
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro")
            # try:
            #     roc_auc = roc_auc_score(all_labels, all_probs, average="macro", multi_class="ovr")
            # except ValueError:
            #     roc_auc = None
            print(f"[Epoch {epoch}] loss={total_loss/len(train_loader):.4f} — "
                  f"val_acc={acc:.4f}, f1={f1:.4f}")

        # save weights if cần
        if model_save_path:
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(model_save_path, "model.pt"))

    def predict_proba(self, texts, batch_size: int = 16):
        """Trả về mảng (n_samples, num_classes) xác suất."""
        self.model.eval()
        enc = self.encode(texts)
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
        loader = DataLoader(ds, batch_size=batch_size)
        probs_list = []
        with torch.no_grad():
            for ids, masks in loader:
                ids, masks = ids.to(self.device), masks.to(self.device)
                logits = self.model(input_ids=ids, attention_mask=masks).logits
                # probs_list.append(F.softmax(logits, dim=-1).cpu().numpy())
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                if probs.ndim == 1:  # Nếu shape là (3,) thì reshape về (1, 3)
                    probs = probs.reshape(1, -1)
                probs_list.append(probs)
        return np.vstack(probs_list)

    def predict(self, texts, batch_size: int = 16):
        probs = self.predict_proba(texts, batch_size)
        return probs.argmax(axis=1)

    def evaluate(self, texts, labels, batch_size: int = 16):
        """Trả về dict gồm accuracy, f1_macro."""
        y_true = np.array(labels)
        probs = self.predict_proba(texts, batch_size)
        y_pred = probs.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        # try:
        #     roc_auc = roc_auc_score(y_true, probs, average="macro", multi_class="ovr")
        # except ValueError:
        #     roc_auc = None
        return {"accuracy": acc, "f1_macro": f1}


# === PyFunc wrapper để MLflow Serving / Flask ===

class DistilBertPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # context.artifacts["model_weights"] trỏ tới file model.pt
        import torch
        self.wrapper = PTDistilBertClassifier(num_classes=context.model_config["num_classes"])
        state = torch.load(context.artifacts["model_weights"], map_location=self.wrapper.device)
        self.wrapper.model.load_state_dict(state)

    def predict(self, context, model_input):
        # model_input: pandas.DataFrame có cột "text"
        texts = model_input["text"].tolist()
        probs = self.wrapper.predict_proba(texts)
        return probs  # MLflow sẽ serialise thành list of lists

