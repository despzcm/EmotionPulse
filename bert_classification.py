import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.optim import AdamW
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score
from enum import Enum
class EmoNum(Enum):
    Emo28 = 28
    Emo7 = 7
    Emo3 = 3
class CustomClassifier(nn.Module):
    def __init__(self, bert_hidden_size, num_labels, dropout_rate=0.3):
        super(CustomClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        # 第一层：768 -> 512
        self.linear1 = nn.Linear(bert_hidden_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.activation1 = nn.GELU()
        
        # 第二层：512 -> 256
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.activation2 = nn.GELU()
        
        # 最后的分类层：256 -> num_labels
        self.classifier = nn.Linear(256, num_labels)
        
    def forward(self, bert_output):
        
        x = self.dropout(bert_output)
        
        # 第一层
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.dropout(x)
        
        # 分类层
        x = self.classifier(x)
        return x

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("正在对文本进行tokenization...")
        encodings = tokenizer(
            [str(text) for text in texts],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        print("tokenization完成！")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class BERTClassifier:
    def __init__(self, num_labels, model_name='google-bert/bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        
        self.bert = BertModel.from_pretrained(model_name).to(self.device)
        
        
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < 9:  # 冻结前9层
                for param in layer.parameters():
                    param.requires_grad = False
            else:  # 解冻最后3层
                for param in layer.parameters():
                    param.requires_grad = True
        
        # 使用自定义分类器
        self.classifier = CustomClassifier(
            bert_hidden_size=768,  
            num_labels=num_labels,
            dropout_rate=0.3  
        ).to(self.device)
        
        
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        bert_params = []
        classifier_params = []
        
        
        for name, param in self.bert.named_parameters():
            if param.requires_grad:  # 只包含解冻的参数
                bert_params.append(param)
                
        for param in self.classifier.parameters():
            classifier_params.append(param)
        
        
        optimizer = AdamW([
            {'params': bert_params, 'lr': learning_rate},  # BERT层使用较小的学习率
            {'params': classifier_params, 'lr': learning_rate * 5}  # 分类器使用较大的学习率
        ])
        
        for epoch in range(epochs):
            self.bert.train()  
            self.classifier.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                
                bert_outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                cls_output = bert_outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(cls_output)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_f1 = self.evaluate(val_loader)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')
            print(f'Validation F1 score: {val_f1:.4f}')

    def evaluate(self, val_loader):
        self.bert.eval()
        self.classifier.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

            
                bert_outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                cls_output = bert_outputs.last_hidden_state[:, 0, :]

                logits = self.classifier(cls_output)
  
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                threshold=0.5
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算每个类别的指标
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels.numpy(), 
            all_preds.numpy(), 
            average=None
        )
        
        # 打印每个类别的详细指标
        print("\n每个类别的评估指标:")
        print("类别\tPrecision\tRecall\tF1-Score\tSupport")
        print("-" * 60)
        for i in range(len(precision)):
            print(f"{i}\t{precision[i]:.4f}\t{recall[i]:.4f}\t{f1[i]:.4f}\t{support[i]}")
        
        # 计算宏平均和微平均
        macro_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')
        micro_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='micro')
        print("\n宏平均F1分数:", macro_f1)
        print("微平均F1分数:", micro_f1)
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, micro_f1

    def save_model(self, path):
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'bert_state_dict': self.bert.state_dict()
        }, path)
        print(f"模型已保存到: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.bert.load_state_dict(checkpoint['bert_state_dict'])
        print(f"模型已从 {path} 加载")

    def predict(self, text, threshold=0.5, return_cls_output=False):
        self.bert.eval()
        self.classifier.eval()
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            # 获取BERT输出
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_output = bert_outputs.last_hidden_state[:, 0, :]
            
            # 通过自定义分类器
            logits = self.classifier(cls_output)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

        if return_cls_output:
            return preds.cpu().numpy()[0], probs.cpu().numpy()[0], cls_output.cpu()
        else:
            return preds.cpu().numpy()[0], probs.cpu().numpy()[0]
    def predict_tokenized(self, input_ids, attention_mask, threshold=0.5, return_cls_output=False):
        """预测已分词的文本
        
        Args:
            input_ids: 已分词的输入ID张量
            attention_mask: 注意力掩码张量
            threshold: 预测阈值，默认为0.5
            return_cls_output: 是否返回CLS输出，默认为False
            
        Returns:
            如果return_cls_output为True，返回(preds, probs, cls_output)
            否则返回(preds, probs)
        """
        self.bert.eval()
        self.classifier.eval()
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            # 获取BERT输出
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_output = bert_outputs.last_hidden_state[:, 0, :]
            
            # 通过自定义分类器
            logits = self.classifier(cls_output)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
        if return_cls_output:
            return preds.cpu().numpy(), probs.cpu().numpy(), cls_output.cpu()
        else:
            return preds.cpu().numpy(), probs.cpu().numpy()
def prepare_go_emotions_data(use_grouped_emotions=False, emotion_type=EmoNum.Emo28):
    #simplified 7-class emotions
    #0:anger, 1:fear, 2:joy, 3:disgust, 4:sadness, 5:surprise, 6:neutral
    #28emo to 7 emo mapping：
    emo_map_7=[2,2,0,0,2,2,5,5,2,4,0,3,4,2,1,2,4,2,2,1,2,2,5,2,4,4,5,6]
    #28emo to 3 emo mapping：
    emo_map_3=[0,0,1,1,0,0,1,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0,0,0,1,1,0,2]
    train_df = pd.read_csv('dataset/go_emotions-train.csv')
    val_df = pd.read_csv('dataset/go_emotions-validation.csv')
    test_df = pd.read_csv('dataset/go_emotions-test.csv')
    
    def process_labels(labels_str):
        labels_str = labels_str.strip('[]')
        labels = labels_str.split()
        if emotion_type == EmoNum.Emo7:
            return [emo_map_7[int(label)] for label in labels]
        elif emotion_type == EmoNum.Emo3:
            return [emo_map_3[int(label)] for label in labels]
        else:
            return [int(label) for label in labels]

    train_texts = train_df['text'].values
    train_labels = train_df['labels'].apply(process_labels).values
    
    val_texts = val_df['text'].values
    val_labels = val_df['labels'].apply(process_labels).values
    
    test_texts = test_df['text'].values
    test_labels = test_df['labels'].apply(process_labels).values
    
    def labels_to_onehot(labels_list, num_classes=28):
        onehot = np.zeros((len(labels_list), num_classes),dtype=np.float32 )
        for i, labels in enumerate(labels_list):
            for label in labels:
                onehot[i, label] = 1
        return onehot
    
    num_labels = emotion_type.value
    train_labels = labels_to_onehot(train_labels, num_labels)
    val_labels = labels_to_onehot(val_labels, num_labels)
    test_labels = labels_to_onehot(test_labels, num_labels)
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    model_name=r"google-bert/bert-base-uncased"
    emotion_type = EmoNum.Emo3  # 设置为Emo3使用3类情感分类
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = prepare_go_emotions_data(emotion_type=emotion_type)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    num_labels = emotion_type.value
    classifier = BERTClassifier(num_labels,model_name)
    
    train_dataset = GoEmotionsDataset(train_texts, train_labels, tokenizer)
    val_dataset = GoEmotionsDataset(val_texts, val_labels, tokenizer)
    test_dataset = GoEmotionsDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    print("开始训练...")
    classifier.train(train_loader, val_loader, epochs=4, learning_rate=2e-5)
    bs=16
    lr=2e-5
    save_path = f"model/bert_emotion_classifier_bs_{bs}_lr_{lr}_use_grouped_emotions_{emotion_type.value}.pt"
    classifier.save_model(save_path)

    print("\n在测试集上评估...")
    test_loss, test_f1 = classifier.evaluate(test_loader)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")

    print("\n预测示例:")
    test_text = "I'm so happy!"
    pred_labels, probs = classifier.predict(test_text)
    print(f"输入文本: {test_text}")
    print(f"预测的情感标签: {pred_labels}")
    print(f"预测的概率: {probs}")

if __name__ == "__main__":
    main()
