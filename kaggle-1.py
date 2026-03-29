# ===================== 1. 导入必要库 =====================
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# ===================== GPU修改：指定运行设备 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")

# ===================== 2. 加载原始数据 =====================
# 请确保 train.csv/test.csv/sample_submission.csv 在当前目录下
# 如果路径不同，修改引号内的路径即可
train_df = pd.read_csv('D:/learn_pytorch/spaceship-titanic/train.csv')
test_df = pd.read_csv('D:/learn_pytorch/spaceship-titanic/test.csv')
submission_df = pd.read_csv('D:/learn_pytorch/spaceship-titanic/sample_submission.csv')
test_passenger_ids = test_df['PassengerId'].copy()  # 这一行就是变量定义

# 打印原始数据基本信息（方便核对）
print("=== 原始数据基本信息 ===")
print(f"训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")
print(f"训练集缺失值前5列统计:\n{train_df.isnull().sum().head()}")


def preprocess_features(df):
    """
    特征工程函数：拆分有用特征、删除无用特征
    注意：不再删除PassengerId（在外部提前保存）
    """
    # 3.1 拆分 Cabin 为 Deck（甲板）、CabinNum（舱号）、Side（左右舷），仅保留 Deck 和 Side
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(['Cabin', 'CabinNum'], axis=1)  # 删除原始Cabin和无意义的舱号

    # 3.2 拆分 PassengerId 为 Group（分组），计算组大小（一起旅行的人数）
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    df = df.drop(['Group'], axis=1)  # 只删除分组号，保留原始PassengerId（后续会统一删除）

    # 3.3 删除无意义特征（Name对预测无帮助）
    df = df.drop('Name', axis=1)

    return df


# 对训练集和测试集应用特征工程
train_df = preprocess_features(train_df)
test_df = preprocess_features(test_df)

# 统一删除PassengerId（特征工程完成后）
X = train_df.drop(['Transported', 'PassengerId'], axis=1)  # 训练集特征
y = train_df['Transported'].astype(int)  # 训练集标签（True=1, False=0）
X_test = test_df.drop('PassengerId', axis=1)  # 测试集特征（无标签）

print("\n=== 特征工程后数据信息 ===")
print(f"训练集列名: {X.columns.tolist()}")
print(f"训练集形状: {X.shape}, 测试集形状: {X_test.shape}")

# ===================== 4. 分离特征和标签 =====================
# 标签：Transported（是否被传送），转为0/1数值型
X = train_df.drop('Transported', axis=1)  # 训练集特征
y = train_df['Transported'].astype(int)  # 训练集标签（True=1, False=0）
X_test = test_df  # 测试集特征（无标签）

# ===================== 5. 定义预处理管道（处理缺失值+编码+标准化） =====================
# 5.1 划分数值特征和类别特征
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# 5.2 数值特征处理：中位数填充缺失值 + 标准化（神经网络必备）
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 中位数填充（抗异常值）
    ('scaler', StandardScaler())  # 标准化（均值0，方差1）
])

# 5.3 类别特征处理：众数填充缺失值 + OneHot编码（转数值）
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 众数填充
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 忽略测试集可能出现的新类别
])

# 5.4 组合预处理管道（同时处理数值+类别特征）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ===================== 6. 拟合+转换数据（生成最终可用的特征数组） =====================
# 6.1 训练集特征：拟合并转换（toarray()转为numpy数组，astype(float32)适配PyTorch）
X_processed = preprocessor.fit_transform(X).astype(np.float32)
# 6.2 测试集特征：仅转换（不能fit，避免数据泄露）
X_test_processed = preprocessor.transform(X_test).astype(np.float32)
# 6.3 标签转换为float32（适配PyTorch的损失函数）
y_processed = y.values.astype(np.float32).reshape(-1, 1)  # 形状：(样本数, 1)

# ===================== 7. 划分训练集/验证集（用于模型训练+验证） =====================
# stratify=y：保证训练集/验证集的标签分布一致
X_train_processed, X_val_processed, y_train_processed, y_val_processed = train_test_split(
    X_processed, y_processed, test_size=0.2, random_state=42, stratify=y
)

# ===================== 8. 输出最终清洗后的数据信息（方便你核对） =====================
print("\n=== 数据清洗完成！最终输出变量信息 ===")
print(f"训练集特征形状: {X_train_processed.shape} (样本数 × 特征数)")
print(f"验证集特征形状: {X_val_processed.shape}")
print(f"测试集特征形状: {X_test_processed.shape}")
print(f"训练集标签形状: {y_train_processed.shape} (样本数 × 1)")
print(f"验证集标签形状: {y_val_processed.shape}")
print(type(X_train_processed))

# ===================== GPU修改：张量移到指定设备（GPU/CPU） =====================
x_train = torch.tensor(X_train_processed).to(device)
x_val = torch.tensor(X_val_processed).to(device)
y_train = torch.tensor(y_train_processed).to(device)
y_val = torch.tensor(y_val_processed).to(device)
x_test = torch.tensor(X_test_processed).to(device)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_val, y_val)

input_dim = x_train.shape[1]
output_dim = 1

class Transported(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 64)
        self.output = nn.Linear(64, output_dim)
        self.bn1 = nn.BatchNorm1d(64, eps = 1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.bn1(self.linear1(x))
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.linear2(x), negative_slope=0.01)
        x = torch.relu(self.linear3(x))
        x = self.dropout(x)
        x = torch.relu(self.linear4(x))
        x = torch.sigmoid(self.output(x))
        return x

def train(train_dataset, input_dim, output_dim):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # ===================== GPU修改：模型移到指定设备 =====================
    model = Transported(input_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    epochs = 100
    model.train()
    for epoch in range(epochs):
        total_loss, batch_num = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/batch_num:.4f}")

    torch.save(model.state_dict(), './model/spaceship.pth')

def evaluate(test_dataset, input_dim, output_dim):
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # ===================== GPU修改：模型移到指定设备 =====================
    model = Transported(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load('./model/spaceship.pth'))
    model.eval()

    correct = 0
    # ===================== GPU修改：推理时禁用梯度计算（省显存） =====================
    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            y_pred = (y_pred > 0.5).float()
            # ===================== GPU修改：张量移回CPU计算 =====================
            correct += (y_pred.cpu() == y.cpu()).sum().item()

    print(f"Accuracy: {correct/len(test_dataset):.2%}")


def predict_and_submit(x_test, test_passenger_ids, input_dim, output_dim):
    """
    对测试集进行预测，并生成符合Kaggle要求的提交文件
    """
    # 加载训练好的模型
    model = Transported(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load('./model/spaceship.pth'))
    model.eval()

    # 对测试集进行预测
    with torch.no_grad():
        test_preds = model(x_test)
        # 将概率转为布尔值（>0.5为True，否则False），并移回CPU转为numpy数组
        test_preds_bool = (test_preds.cpu().numpy() > 0.5).flatten().astype(bool)

    # 生成提交DataFrame（严格匹配sample_submission格式）
    submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Transported': test_preds_bool
    })

    # 保存提交文件
    submission.to_csv('./submission.csv', index=False)
    print("\n提交文件已生成：./submission.csv")
    print(f"提交文件形状: {submission.shape}")
    print("\n提交文件前5行预览：")
    print(submission.head())



if __name__ == '__main__':
    train(train_dataset, input_dim, output_dim)
    evaluate(test_dataset, input_dim, output_dim)
    predict_and_submit(x_test, test_passenger_ids, input_dim, output_dim)