# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
<img width="2126" height="740" alt="image" src="https://github.com/user-attachments/assets/ef36fe43-fbec-4994-becf-9f69ae827dfc" />


## DESIGN STEPS
### STEP 1: 
Load and Preprocess Data.

### STEP 2: 
Get the shape of the first image in the training dataset.


### STEP 3: 
Get the shape of the first image in the test dataset.


### STEP 4: 
Train and test the Model.


### STEP 5: 
Predict on a Single Image.


### STEP 6: 
Display the image.




## PROGRAM

### Name: Cholimgapuram Sai Likitha

### Register Number:212224230046

```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Name: Cholimgapuram Sai Likiha")
        print("Register Number: 212224230046")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

train_model(model, train_loader, num_epochs=10)

```

### OUTPUT

## Training Loss per Epoch

<img width="501" height="657" alt="image" src="https://github.com/user-attachments/assets/79557c9c-b15f-4c7c-8ea2-a38f2551c4eb" />


## Confusion Matrix

<img width="755" height="682" alt="image" src="https://github.com/user-attachments/assets/89f4eb14-552b-44d7-a395-1a8a31f0147b" />


## Classification Report
<img width="595" height="363" alt="image" src="https://github.com/user-attachments/assets/079c5e2e-b1df-469d-8c27-640b4234a378" />


### New Sample Data Prediction
<img width="521" height="496" alt="image" src="https://github.com/user-attachments/assets/68e83485-cfbf-4b8f-947a-65116a6eac17" />


## RESULT
Thus the output is predicted and displayed
