from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2d = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)
        self.soft_max = nn.Softmax()

        nn.init.kaiming_uniform_(self.cnn_1.weight)  # He init
        nn.init.kaiming_uniform_(self.cnn_2.weight)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        pass

    def forward(self, x):
        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)
        return out

