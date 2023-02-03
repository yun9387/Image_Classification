#識別モデル
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# エポック数
MAX_EPOCH = 3
train_img_dir = r""

# ニューラルネットワークの定義
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,3,5)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.layer1 = nn.Linear(16 * 5 * 5, 120)
        self.layer2 = nn.Linear(120, 84)
        self.layer3 = nn.Linear(84, 10)

    def forward(self, input_data):
        input_data = self.pool(F.relu(self.conv1(input_data)))
        input_data = self.pool(F.relu(self.conv2(input_data)))
        input_data = self.pool(F.relu(self.conv3(input_data)))
        input_data = input_data.view(-1, 16 * 5 * 5)
        input_data = F.relu(self.layer1(input_data))
        input_data = F.relu(self.layer2(input_data))
        input_data = self.layer3(input_data)
        return input_data


def main():

    # transform定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # 学習データ
    train_data = torchvision.datasets.ImageFolder(root=train_img_dir,transform=transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

    model = CNN()

    # optimizerの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    # 学習
    for epoch in range(MAX_EPOCH):
        total_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            # 学習データと教師ラベルデータを取得
            train_data, teacher_labels = data
            # 勾配情報を削除
            optimizer.zero_grad()
            # モデルで予測を計算
            outputs = model(train_data)
            # 微分計算
            loss = criterion(outputs, teacher_labels)
            loss.backward()
            # 勾配を更新
            optimizer.step()
            # 誤差
            total_loss += loss.item()
            # 1000ミニバッチずつ進捗を表示
            if i % 1000 == 999:
                print('学習進捗：[学習回数：%d, ミニバッチ数：%5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / 1000))
                total_loss = 0.0
                
    # モデル保存
    torch.save(model.state_dict(), "model.pth")

    print("-----学習完了-----")

if __name__ == "__main__":
    main()
