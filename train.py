import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    df = pd.read_csv(path + "/sign_mnist_train.csv")

    y = df['label'].values
    X = df.drop('label', axis=1).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_train = (X_train - 127.5) / 127.5 
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, 26) 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="Sign_Language_Run")
    args = parser.parse_args()

    # READS THE URI FROM YOUR GITHUB SECRETS
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    dataset = load_and_preprocess_data()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    mock_accuracy = os.getenv("MOCK_ACCURACY")

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({"learning_rate": args.learning_rate, "batch_size": args.batch_size, "epochs": args.epochs})
        mlflow.set_tag("student_id", "202201623")

        for epoch in range(args.epochs):
            total_loss, correct, total = 0, 0, 0
            
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total

            mlflow.log_metric("loss", avg_loss, step=epoch)
            
            if mock_accuracy is not None and epoch == args.epochs - 1:
                accuracy = float(mock_accuracy)
                print(f"Mocking final accuracy to {accuracy} for testing.")

            mlflow.log_metric("accuracy", accuracy, step=epoch)

        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    main()