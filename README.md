# Textclass4bert
# Text classification for BERT

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define a custom dataset class for text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = [self.preprocess_text(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer

    @staticmethod
    def preprocess_text(text):
        text = text.lower()  # convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
        tokens = nltk.word_tokenize(text)  # tokenize
        tokens = [t for t in tokens if t not in stopwords.words('english')]  # remove stopwords
        tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]  # lemmatization
        return ' '.join(tokens)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Encode the text to be suitable for BERT
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define the BERT model for text classification
class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

# Main training function
def train(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

# Evaluate the model on the validation set
def evaluate(model, data_loader, loss_fn, device):
    model = model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            total_correct += torch.sum(preds == labels)
            total_samples += len(labels)
    accuracy = total_correct.double() / total_samples
    return accuracy

# Prepare for training
def prepare_data(args):
    # Load data
    df = pd.read_csv(args.data_path)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Create data loaders
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data_loader = DataLoader(
        TextDataset(train_df.text.to_numpy(), train_df.label.to_numpy(), tokenizer),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_data_loader = DataLoader(
        TextDataset(val_df.text.to_numpy(), val_df.label.to_numpy(), tokenizer),
        batch_size=args.batch_size,
        shuffle=False
    )
    return train_data_loader, val_data_loader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_loader, val_data_loader = prepare_data(args)

    model = BertClassifier(n_classes=args.num_classes)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_data_loader) * args.epochs
    scheduler = StepLR(optimizer, step_size=total_steps//3, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss().to(device)

    best_accuracy = 0

    for epoch in range(args.epochs):
        logging.info(f'Epoch {epoch + 1}/{args.epochs}')
        logging.info('-' * 10)

        train_acc, train_loss = train(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_data_loader.dataset)
        )

        logging.info(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc = evaluate(
            model,
            val_data_loader,
            loss_fn,
            device
        )

        logging.info(f'Val accuracy {val_acc}')

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), args.model_path)
            best_accuracy = val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--model_path', type=str, default='bert_model.bin', help='Path to save the trained model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes for classification')
    args = parser.parse_args()
    main(args)

