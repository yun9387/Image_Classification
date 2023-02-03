#分類
import torch
import torchvision
import torchvision.transforms as transforms
from torch_sample import CNN

test_img_dir = r""

def main():

    # モデル読み込み
    model = CNN()
    model.load_state_dict(torch.load("model.pth"))

    # transform定義
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 検証データ
    test_data = torchvision.datasets.ImageFolder(root=test_img_dir,trnsform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

    # クラスの中身を設定
    class_names = ('', '', '', '', '', '', '', '', '', '')

    # クラスごとの検証結果
    class_corrent = list(0. for i in range(10))
    class_total = list(0.for i in range(10))

    with torch.no_grad(): # 勾配の計算をしない
        for data in test_data_loader:
            # 検証データと教師ラベルデータを取得
            test_data, teacher_labels = data
            # 検証データをモデルに渡し予測
            results = model(test_data)
            # 予測結果を取得
            _, predicted = torch.max(results, 1) 
            c = (predicted == teacher_labels).squeeze()
            for i in range(4):
                label = teacher_labels[i]
                class_corrent[label] += c[i].item()
                class_total[label] += 1
    # 結果表示
    for i in range(10):
        print(' %5s クラスの正解率: %2d %%' % (class_names[i], 100 * class_corrent[i] / class_total[i]))
        
if __name__ == "__main__":
    main()