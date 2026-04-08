import re
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
#!wget https://raw.githubusercontent.com/cbannard/lela60331_24-25/refs/heads/main/coursework/Compiled_Reviews.txt

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_reviews(file):
    """Reads the reviews and sentiment ratings from a file."""
    reviews=[]
    sentiment_ratings=[]
    with open(file) as f:
        for line in f.readlines()[1:]:
            fields = line.rstrip().split('\t')
            reviews.append(fields[0])
            sentiment_ratings.append(fields[1])
    return reviews, sentiment_ratings

def pre_process(reviews, stop_words):
    """Preprocesses the reviews by lowercasing, removing punctuation, tokenizing, and removing stop words. It also creates a vocabulary mapping."""
    preprocess = [re.sub(r"[,.;:?!()\"\]\[]+\ *", ' ', txt.lower()).split() for txt in reviews]
    preprocess = [[word for word in txt if word not in stop_words] for txt in preprocess]
    vocab_list = list(set([word for txt in preprocess for word in txt]))
    return {word: index+1 for index, word in enumerate(vocab_list)}

def encode_word2int(data, stop_words, vocab):
    """Encodes the reviews into sequences of integers based on the vocabulary mapping."""
    word2int = []
    for text in data:
        tokens = re.sub(r"[,.;:?!()\"\]\[]+\ *", ' ', text.lower()).split()
        tokens = [word for word in tokens if word not in stop_words]
        word2int.append([vocab.get(word, 0) for word in tokens])
    return word2int

def x_y(encoded_train_data, sentiment_ratings, MAX_SEQ_LEN):
    """Pads or truncates the encoded reviews to a fixed length and creates the corresponding labels. Calculates lengths."""
    X = []
    lengths = []
    for sentence in encoded_train_data:
        if len(sentence) > MAX_SEQ_LEN:
            X.append(sentence[:MAX_SEQ_LEN])
            lengths.append(MAX_SEQ_LEN)
        else:
            X.append(sentence + [0]*(MAX_SEQ_LEN-len(sentence)))
            lengths.append(len(sentence))
    x = torch.tensor(X, dtype=torch.long)
    y = [1 if label == "positive" else 0 for label in sentiment_ratings]
    return x, y, torch.tensor(lengths)

def t_t_s(x, y, lengths):
    """Splits the data into 80% training, 10% validation, and 10% test sets. Lengths also included for later use in LSTM padding."""
    x_train, x_other, y_train, y_other, len_train, len_other = train_test_split(x, y, lengths, test_size=0.2, random_state=30)
    x_val, x_test, y_val, y_test, len_val, len_test = train_test_split(x_other, y_other, len_other, test_size=0.5, random_state=30)
    return x_train, y_train, x_val, y_val, x_test, y_test, len_train, len_val, len_test

def data_loader(x_train, y_train, x_val, y_val, x_test, y_test, len_train, len_val, len_test):
    """Creates DataLoader for the training set and TensorDatasets for the validation and test sets. Lengths also included."""
    train_set = TensorDataset(x_train, torch.tensor(y_train, dtype=torch.long), len_train)
    val_set = TensorDataset(x_val, torch.tensor(y_val, dtype=torch.long), len_val)
    test_set = TensorDataset(x_test, torch.tensor(y_test, dtype=torch.long), len_test)

    train_loader = DataLoader(train_set, batch_size=64, pin_memory=True, shuffle=True)
    return train_loader, val_set, test_set

class logistic_regression(nn.Module):
    """Defines a logistic regression model with an embedding layer (so the comparison is between model architecture not inputs) and a linear layer. Masks the embeddings to account for padding."""
    def __init__(self, vocab_size, embedding_dim):
        super(logistic_regression, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.layer_1 = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1)
        emb = emb * mask
        x = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.sigmoid(self.layer_1(x))

class multilayer_network(nn.Module):
    """Defines a multilayer perceptron with an embedding layer, two linear layers, ReLU activation, and dropout. Masks the embeddings to account for padding."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(multilayer_network, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(0.1)
        self.fc_1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc_2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1)
        emb = emb * mask
        summed = emb.sum(dim=1) 
        lengths = mask.sum(dim=1).clamp(min=1)
        x = summed / lengths
        x = self.dropout1(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return torch.sigmoid(self.fc_2(x))
    
class LSTM(nn.Module):
    """Defines an LSTM model with an embedding layer, an LSTM layer, a linear layer, dropout, and sigmoid activation. Compresses out padding for efficiency.
    Similar structure to the MLP for closer model comparison: same embedding_dim=128, hidden_dim=64, num_layers=2, dropout=[0.1,0.2], padding removed."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=0.1, num_layers = num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, lengths, h):
        embeds = self.embedding(x)
        packed = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed, h)
        lstm_out = h_n[-1]
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return torch.sigmoid(out)
  
def trainer(model, train_loader, optimizer, device, x_val, y_val, epochs):
    """Trains the Logistic Regression and MLP models and implements early stopping based on the validation loss."""
    loss_by_epoch=[]
    min_loss=99999
    for epoch in range(epochs):
        model.train()
        cum_loss=0.0
        for texts, labels, _ in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            preds = model(texts)
            loss = nn.BCELoss()(preds.squeeze(-1), labels.float())
            optimizer.zero_grad()
            loss.backward()
            cum_loss+=loss.item()
            optimizer.step()
        loss_by_epoch.append(cum_loss)
        model.eval()
        with torch.no_grad():
            val_texts = x_val.to(device)
            val_labels = torch.tensor(y_val, dtype=torch.float32, device=device)
            val_preds = model(val_texts).squeeze(-1)
            loss_val = nn.BCELoss()(val_preds, val_labels).item()
            if loss_val <= min_loss:
                min_loss = loss_val
            elif (loss_val/min_loss) > 1.1:
                break
    return loss_by_epoch

def LSTM_trainer(model, train_loader, optimizer, device, x_val, y_val, len_val, epochs, num_layers, hidden_dim):
    """Trains the LSTM model and implements early stopping based on the validation loss."""
    loss_by_epoch=[]
    min_loss=99999
    for epoch in range(epochs):
        model.train()
        cum_loss=0.0
        for texts, labels, lengths in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            h0 = torch.zeros(num_layers, labels.shape[0], hidden_dim, device=device)
            c0 = torch.zeros(num_layers, labels.shape[0], hidden_dim, device=device)
            h_c = (h0, c0)
            preds = model(texts, lengths, h_c)
            loss = nn.BCELoss()(preds.squeeze(-1), labels.float())
            optimizer.zero_grad()
            loss.backward()
            cum_loss+=loss.item()
            optimizer.step()
        loss_by_epoch.append(cum_loss)
        model.eval()
        with torch.no_grad():
            h0 = torch.zeros(num_layers, len(x_val), hidden_dim, device=device)
            c0 = torch.zeros(num_layers, len(x_val), hidden_dim, device=device)
            h_c = (h0, c0)
            preds_val=model(x_val.to(device), len_val, h_c)
            loss_val = nn.BCELoss()(preds_val.squeeze(-1), torch.from_numpy(np.array(y_val)).float().to(device)).item()
            if loss_val <= min_loss:
                min_loss = loss_val
            elif (loss_val/min_loss) > 1.1:
                break
    return loss_by_epoch

def get_probs(model, x, device):
    """Gets the predicted probabilities from the model for the given input."""
    model.eval()
    with torch.no_grad():
        return (model(x.to(device)).squeeze(-1).cpu()).tolist()
    
def get_probs_LSTM(model, x, lengths, device, num_layers, hidden_dim):
    """Gets the predicted probabilities from the LSTM model for the given input."""
    model.eval()
    h0 = torch.zeros(num_layers, len(x), hidden_dim, device=device)
    c0 = torch.zeros(num_layers, len(x), hidden_dim, device=device)
    h_c = (h0, c0)
    with torch.no_grad():
        return (model(x.to(device), lengths, h_c).squeeze(-1).cpu()).tolist()

def evaluate_probs(probs, y_true):
    """Evaluates the predicted probabilities against the true labels and returns the F1 score."""
    preds = [int(p > 0.5) for p in probs]
    return precision_recall_fscore_support(y_true, preds, average="macro")[2]

def plot_training_loss(loss1, loss2, loss3):
    plt.figure()
    plt.plot(loss1, label="Logistic Regression")
    plt.plot(loss2, label="Multilayer Network")
    plt.plot(loss3, label="LSTM")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    return plt.savefig("training_loss.png")

def plot_density(preds_and_labels, model):
    plt.figure()
    preds_and_labels[preds_and_labels["lab"] == 0]["p"].plot(kind="density")
    preds_and_labels[preds_and_labels["lab"] == 1]["p"].plot(kind="density")
    plt.xlabel(f"Predicted Probability: {model}")
    plt.title(f"Prediction Distribution by Class: {model}")
    return plt.savefig(f"density_plot_{model}.png")

def plot_roc_curve(y_test,probs,model_name):
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic: {model_name}')
    return plt.savefig(f"roc_curve_{model_name}.png")

def get_auc(y, probs):
    """Calculates the AUC for the given true labels and predicted probabilities."""
    fpr, tpr, thresholds = roc_curve(y, probs)
    return auc(fpr, tpr)

def bootstrap_auc(y, probs, model, num_samples):
    """Performs a bootstrap test to estimate the confidence interval for the AUC of a model."""
    model.eval()
    scores = []
    for i in range(num_samples):
        indices = np.random.choice(len(y), len(y), replace=True)
        fpr, tpr, thresholds = roc_curve(np.array(y)[indices], np.array(probs)[indices])
        scores.append(auc(fpr, tpr))
    scores = np.sort(scores)
    return (np.mean(scores),scores[math.floor(len(scores)*0.025)],scores[math.floor(len(scores)*0.975)])

def bootstrap_p_value(y, prob_1, prob_2, num_samples):
    """Performs a bootstrap test to compare the AUC of two models and returns the p-value for the hypothesis that model 2 has a higher AUC than model 1.
    A p-value < 0.05 would indicate that model 2 has a significantly higher AUC than model 1."""
    diffs = []
    for i in range(num_samples):
        idx = np.random.choice(len(y), len(y), replace=True)
        auc_1 = auc(*roc_curve(np.array(y)[idx], np.array(prob_1)[idx])[:2])
        auc_2 = auc(*roc_curve(np.array(y)[idx], np.array(prob_2)[idx])[:2])
        diffs.append(auc_2 - auc_1)
    diffs = np.array(diffs)
    p_value = np.mean(diffs <= 0)
    return p_value

if __name__ == "__main__":
    """Main function to execute the entire workflow: data loading, preprocessing, model training, evaluation, and result saving."""
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    MAX_SEQ_LEN = 256
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
   
    reviews, sentiment_ratings = get_reviews("Compiled_Reviews.txt")

    reviews_filtered = []
    sentiment_filtered = []
    for r, s in zip(reviews, sentiment_ratings):
        tokens = re.sub(r"[,.;:?!()\"\]\[]+\ *", ' ', r.lower()).split()
        tokens = [word for word in tokens if word not in stop_words]
        if tokens:
            reviews_filtered.append(r)
            sentiment_filtered.append(s)
    reviews = reviews_filtered
    sentiment_ratings = sentiment_filtered

    vocab = pre_process(reviews, stop_words)
    encoded_train_data = encode_word2int(reviews, stop_words, vocab)
    x, y, lengths = x_y(encoded_train_data, sentiment_ratings, MAX_SEQ_LEN)

    x_train, y_train, x_val, y_val, x_test, y_test, len_train, len_val, len_test = t_t_s(x, y, lengths)
    train_loader, val_set, test_set = data_loader(x_train, y_train, x_val, y_val, x_test, y_test, len_train, len_val, len_test)

    vocab_size = len(vocab) + 1
    embedding_dim = 128
    hidden_dim = 64
    num_layers = 2
    epochs = 50
    lr = 0.0001
    
    model_1 = logistic_regression(vocab_size, embedding_dim).to(device)
    optimizer_1 = torch.optim.Adam(params = model_1.parameters(), lr=lr)
    model_2 = multilayer_network(vocab_size, embedding_dim, hidden_dim).to(device)
    optimizer_2 = torch.optim.Adam(params = model_2.parameters(), lr=lr)
    model_3 = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    optimizer_3 = torch.optim.Adam(params = model_3.parameters(), lr=lr)

    loss1 = trainer(model_1, train_loader, optimizer_1, device, x_val, y_val, epochs)
    loss2 = trainer(model_2, train_loader, optimizer_2, device, x_val, y_val, epochs)
    loss3 = LSTM_trainer(model_3, train_loader, optimizer_3, device, x_val, y_val, len_val, epochs, num_layers, hidden_dim)
    plot_training_loss(loss1, loss2, loss3)

    preds_and_labels=pd.DataFrame({"p" : (model_1(x_test.to(device)).squeeze(-1).cpu()).tolist(), "lab" : y_test})
    plot_density(preds_and_labels, "Logistic Regression")

    preds_and_labels2=pd.DataFrame({"p" : (model_2(x_test.to(device)).squeeze(-1).cpu()).tolist(), "lab" : y_test})
    plot_density(preds_and_labels2, "Multilayer Network")

    h0 = torch.zeros(2, len(x_test), hidden_dim).to(device)
    c0 = torch.zeros(2, len(x_test), hidden_dim).to(device)
    preds_and_labels3=pd.DataFrame({"p" : (model_3(x_test.to(device), len_test, (h0, c0)).squeeze(-1).cpu()).tolist(), "lab" : y_test})
    plot_density(preds_and_labels3, "LSTM")

    probs1 = get_probs(model_1, x_test, device)
    probs2 = get_probs(model_2, x_test, device)
    probs3 = get_probs_LSTM(model_3, x_test, len_test, device, num_layers, hidden_dim)

    plot_roc_curve(y_test, probs1, "Logistic Regression")
    plot_roc_curve(y_test, probs2, "Multilayer Network")
    plot_roc_curve(y_test, probs3, "LSTM")

    auc_1 = get_auc(y_test, probs1)
    auc_2 = get_auc(y_test, probs2)
    auc_3 = get_auc(y_test, probs3)

    num_samples = 2000

    result = {'Logistic Regression: F1 Test': evaluate_probs(probs1, y_test),
              'Multilayer Network: F1 Test': evaluate_probs(probs2, y_test),
              'LSTM: F1 Test': evaluate_probs(probs3, y_test),
              'Logistic Regression: AUC': auc_1,
              'Multilayer Network: AUC': auc_2,
              'LSTM: AUC': auc_3,
              'Logistic Regression: Bootstrap AUC': bootstrap_auc(y_test, probs1, model_1, num_samples),
              'Multilayer Network: Bootstrap AUC': bootstrap_auc(y_test, probs2, model_2, num_samples),
              'LSTM: Bootstrap AUC': bootstrap_auc(y_test, probs3, model_3, num_samples),
              'Multilayer Network > Logistic Regression if Bootstrap p-value < 0.05': bootstrap_p_value(y_test, probs1, probs2, num_samples),
              'LSTM > Logistic Regression if Bootstrap p-value < 0.05': bootstrap_p_value(y_test, probs1, probs3, num_samples)
    }
    results = []
    results.append(result)

    df = pd.DataFrame(results)
    df.to_csv('trgs_results.csv', index=False)
