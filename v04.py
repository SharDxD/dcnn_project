# Importing the necessary libraries
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
from torchsummary import summary
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch.optim as optim
from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
torch.manual_seed(0)


###################

# Downloading and Loading FashionMNIST dataset

transform = transforms.Compose([transforms.ToTensor()])


train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


# train_data = train_data.data.numpy()
# test_data = test_data.data.numpy()

training_images = np.zeros((len(train_data), 28, 28))
training_labels = np.zeros(len(train_data), dtype=int)

for i, (img, label) in enumerate(train_data):
    training_images[i] = img.numpy()
    training_labels[i] = int(label)

testing_images = np.zeros((len(test_data), 28, 28))
testing_labels = np.zeros(len(test_data), dtype = int)

for i, (img, label) in enumerate(test_data):
    testing_images[i] = img.numpy()
    testing_labels[i] = int(label)
print('Training Data: X=%s, y=%s' % (training_images.shape, training_labels.shape))
print('Testing Data : X=%s, y=%s' % (testing_images.shape, testing_labels.shape))
print('(Data is normalized between 0 and 1 because of transform function)')

###################

# Data Cleansing
t1 = np.isnan(training_images).any()
t2 = np.isnan(training_labels).any()

if(not t1 and not t2):
	print('Data is clean')

X_train_all = training_images.astype('float32')
X_test  = testing_images.astype('float32')

####################

# Splitting the training dataset to train and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X_train_all, training_labels, stratify=training_labels, test_size=0.2, random_state=42)

print("Splitting the training dataset into training and validation (80% Train and 20% Validation)")
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('val  : X=%s, y=%s' % (X_val.shape, y_val.shape))
print('Test : X=%s, y=%s' % (X_test.shape, testing_labels.shape))

####################

# Flatten the images to 1-D by reshaping
X_train_all = X_train_all.reshape(60000,784)
X_train = X_train.reshape(48000,784)
X_val   = X_val.reshape(12000,784)
X_test  = X_test.reshape(10000,784)

print("Training and validation matrix combined shape  :", X_train_all.shape)
print("Training matrix shape  :", X_train.shape)
print("Validation matrix shape:", X_val.shape)
print("Test matrix shape      :", X_test.shape)


# Performing PCA on both training and validation data combined
from sklearn.decomposition import PCA
pca_all = PCA(n_components = 784)
# Fit on training data
pca_fit = pca_all.fit_transform(X_train_all)
#percentage variance explained
ratio_variance = np.cumsum(pca_all.explained_variance_ratio_)
pca_fit.shape

####################

pca = PCA(n_components=0.95)
pca.fit(X_train)
pca.n_components_

####################

label_mapping = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

def map_to_label(value):
    return label_mapping.get(int(value), 'Unknown')

# Apply the mapping function element-wise to the array
labels = np.vectorize(map_to_label)(training_labels)

#####################

# Reshaping the Data into original size
X_train_t = pca.transform(X_train)
X_val_t   = pca.transform(X_val)
X_test_t  = pca.transform(X_test)

X_train_i = pca.inverse_transform(X_train_t)
X_val_i   = pca.inverse_transform(X_val_t)
X_test_i  = pca.inverse_transform(X_test_t)


training_images         = X_train.reshape(48000,28,28)
reduced_training_images = X_train_i.reshape(48000,28,28)


###########################

from sklearn.manifold import TSNE

# Invoking the t-SNE method
tsne = TSNE(n_components=3)
tsne_dt = X_train_all[:5000]
tsne_results = tsne.fit_transform(tsne_dt)


##########################


# Building the Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Adjust dropout rate as needed
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.cnn = nn.Sequential(
            self.make_cnn_block(1, 16, 3),
            nn.MaxPool2d(2,2),
            self.make_cnn_block(16, 32, 3),
            self.make_cnn_block(32, 64, 3),
            nn.MaxPool2d(2,2),
            self.make_cnn_block(64, 128, 3,final_layer = True),
        )
        self.fnn = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10)
        )


    def make_cnn_block(self, input_channels, output_channels, kernel_size=3, stride=1, padding = 0, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding = padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.relu(x)
    #     x = self.maxpool1(x)
    #     x = self.conv2(x)
    #     x = self.relu(x)
    #     x = self.maxpool2(x)
    #     x = x.view(x.size(0), -1)  # Flatten
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x = self.fc2(x)
    #     return x

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)  # Apply dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
# New functions here
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', 
                               restore_best_weights=True)

def monte_carlo_prediction(model, input, num_samples=100):
    model.train()  # Keep the model in training mode to use dropout
    predictions = [model(input) for _ in range(num_samples)]
    return torch.stack(predictions)

def softmax_with_temperature(logits, temperature=1.0):
    e_x = torch.exp(logits / temperature)
    return e_x / e_x.sum(dim=-1, keepdim=True)

def is_ood(softmax_probs, threshold=0.8):
    max_probs, _ = torch.max(softmax_probs, dim=1)
    return max_probs < threshold
#####################

# Creating TensorDatasets and DataLoaders
X_train_cnn = torch.tensor(X_train.reshape(-1,1,28,28)).float()
y_train_cnn = torch.tensor(y_train).long()
X_val_cnn   = torch.tensor(X_val.reshape(-1,1,28,28)).float()
y_val_cnn   = torch.tensor(y_val).long()
X_test_cnn  = torch.tensor(X_test.reshape(-1,1,28,28)).float()
y_test_cnn  = torch.tensor(testing_labels).long()


train_dataset = TensorDataset(X_train_cnn, y_train_cnn)
val_dataset   = TensorDataset(X_val_cnn, y_val_cnn)
test_dataset  = TensorDataset(X_test_cnn, y_test_cnn)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min',     ###########
                               restore_best_weights=True)
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#####################

model = CNN()
# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_accuracy = 0.0

#####################

#Training the model
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    correct_train_preds = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_train_preds += (predicted == labels).sum().item()

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    train_accuracy = correct_train_preds / len(train_dataset)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct_val_preds = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            running_val_loss += val_loss.item() * val_inputs.size(0)

            _, predicted = torch.max(val_outputs, 1)
            correct_val_preds += (predicted == val_labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    val_accuracy = correct_val_preds / len(val_dataset)
    val_accuracies.append(val_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_cnn_model.pth')

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Training Loss: {epoch_train_loss:.4f}, '
          f'Validation Loss: {epoch_val_loss:.4f}, '
          f'Training Accuracy: {train_accuracy:.4f}, '
          f'Validation Accuracy: {val_accuracy:.4f}')

# Plotting training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()


######################

#Evaluating of best model on test dataset
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics


# Initialize your Convolution neural network class
model = CNN()
# Load the saved best model weights
model.load_state_dict(torch.load('best_cnn_model.pth'))

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def plot_learning_curves(history):            
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

###############################################plot_learning_curves(history)

# Set the model to evaluation mode

# model.eval()
# with torch.no_grad():

#     for images, labels in test_loader:
#         mc_predictions = monte_carlo_prediction(model, images, num_samples=50)
        
    # for images, _ in test_loader:
    #     logits = model(images)
    #     softmax_probs = softmax_with_temperature(logits, temperature=1.0)
    #     ood_flags = is_ood(softmax_probs, threshold=0.8)
    #     print("OOD Flags:", ood_flags)

model.eval()

predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

#Compute confusion matrix
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual labels')
plt.xlabel('Predicted labels')
plt.show()

print(classification_report(true_labels, predictions))

#Calculate the confusion matrix and accuracy
# conf_matrix = confusion_matrix(true_labels, predictions)
# print("Classification report for best CNN on test dataset :\n%s\n"
#       % (metrics.classification_report(true_labels, predictions)))

# print("\nConfusion Matrix on Test dataset:")
# print(conf_matrix)