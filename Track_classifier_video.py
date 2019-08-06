import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch import onnx
from CNN_video import CNN
import datetime
import numpy as np
import torch
now = datetime.datetime.now()
DEVICE = torch.device('cuda:0')
DATASET_PATH = '/media/aigul/Tom/Aigul/Wide_field/tracks_output_verified'
IMG_SIZE = 19
VIDEO_SEQ_LEN = 4
NEG_POS_RATIO = 3.0
TEST_DATASET_SHARE = 0.1
import scipy.io as sio
loaded_mat = sio.loadmat(DATASET_PATH + '/full_19x19.mat')
X_full = loaded_mat['X']
y_full = loaded_mat['y'].astype(np.int64).squeeze()
X_full = np.divide(X_full, 255.0)
X_full = np.subtract(X_full, 0.5)
print("Full dataset: ", X_full.shape, y_full.shape)
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.1, random_state=42)
print("Train dataset: ", X_train.shape, y_train.shape)
print("Val dataset: ", X_val.shape, y_val.shape)
train_batch_size = 128
val_batch_size = 128
X_train = torch.from_numpy(X_train)
X_val = torch.from_numpy(X_val)
Y_train = torch.from_numpy(y_train).type(torch.LongTensor)
Y_val = torch.from_numpy(y_val).type(torch.LongTensor)
train = torch.utils.data.TensorDataset(X_train, Y_train)
val = torch.utils.data.TensorDataset(X_val, Y_val)
train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val, batch_size=val_batch_size, shuffle=False)

model = CNN()
model = model.double()
model.type(torch.cuda.DoubleTensor)
model.to(DEVICE)
LR = 0.0001
L2_reg = 0.05
criterion = nn.CrossEntropyLoss().type(torch.cuda.DoubleTensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
epochs = 150
train_losses, val_losses = [], []
accuracy_train = []
f1 = []
precision = []
recall = []
for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        train = Variable(images.view(-1, 4, 19, 19))
        labels = Variable(labels)

        optimizer.zero_grad()

        output = model(train)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        val_loss = 0
        accuracy = 0
        labels_array = np.array([])
        top_class_array = np.array([])

        with torch.no_grad():  # Turning off gradients to speed up
            model.eval()
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                val = Variable(images.view(-1, 4, 19, 19))
                labels = Variable(labels)
                labels_array = np.hstack([labels_array, np.asarray(labels.type(torch.DoubleTensor))])
                log_ps = model(val)
                val_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                top_class_array = np.hstack([top_class_array, np.asarray(top_class.type(torch.DoubleTensor)).flatten()])
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        model.train()
        f1_epoch = metrics.f1_score(labels_array, top_class_array)
        precision_epoch = metrics.precision_score(labels_array, top_class_array)
        recall_epoch = metrics.recall_score(labels_array, top_class_array)
        f1.append(f1_epoch)
        precision.append(precision_epoch)
        recall.append(recall_epoch)
        accuracy_train.append(accuracy / len(val_loader))
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
              "Val Loss: {:.3f}.. ".format(val_loss / len(val_loader)),
              "Val Accuracy: {:.3f}".format(accuracy / len(val_loader)),
              "F1-score: {:.3f}".format(f1_epoch),
              "Precision: {:.3f}".format(precision_epoch),
              "Recall: {:.3f}".format(recall_epoch))

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].plot(train_losses, label='Training loss')
ax[0].plot(val_losses, label='Validation loss')
ax[0].legend(frameon=False)
ax[1].plot(accuracy_train, label='Accuracy')
ax[1].plot(f1, label='F1-score')
ax[1].plot(precision, label='Precision')
ax[1].plot(recall, label='Recall')
ax[1].legend(frameon=False)
plt.savefig('CNN_models_saved/Loss_Accuracy_' + now.strftime("%d_%m_%Y_%H:%M") + '.png')
plt.show()

path_to_torch_model = '/media/aigul/Tom/Aigul/Wide_field/ML_DL_wide_field/CNN_models_saved/model_video_' + now.strftime("%d_%m_%Y_%H:%M") + '.torch'
torch.save(model.state_dict(), path_to_torch_model)
model_torch = CNN()
model_torch.load_state_dict(torch.load(path_to_torch_model))
x = torch.randn(1, 4, 19, 19).float()
onnx.export(model_torch, x, path_to_torch_model[:-5] + 'onnx', input_names=['input'], output_names=['output'])
train_batch_size = 128
val_batch_size = 128





