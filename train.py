from transformers import *
from datasets import *
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
import torch
import nsml

def bind_model(model, **kwargs):
    def save(filename, **kwargs):
        torch.save(model.state_dict(), f"{filename}/checkpoint.pt")
        print(f"Model saved at : {filename}/checkpoint.pt")

    def load(filename):
        checkpoint = torch.load(f"{filename}/checkpoint.pt")
        model.load_state_dict(checkpoint)
        print(f"Model named {filename} loaded")

    def infer(raw_data):  # TODO: Not complete(see below)
        data = raw_data  # TODO: need to convert raw_data to torch.Tensor object, which can be then fed into model.forward()
        model.eval()
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        predicted = np.array(predicted.cpu())
        print(f"Predicted as: {predicted}")
        return predicted

    nsml.bind(save=save, load=load, infer=infer)


# for korean: https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP

class TwitterDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset['input_ids']
        self.attn_mask = dataset['attention_mask']
        self.ttids = dataset['token_type_ids']
        self.label_list = dataset['sentiment']
        self.label_list = (self.label_list - 2) // 2

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        inputs = {'input_ids': self.input_ids[idx], 'attention_mask': self.attn_mask[idx], 'token_type_ids': self.ttids[idx]}
        return inputs, self.label_list[idx]

class Trainer:
    def __init__(self):
        self.args = {'epoch': 10, 'learning_rate': 5e-05}
        self.step = 0
        try:
            from nsml import DATASET_PATH
            import nsml
            self.nsml = True
        except:
            self.nsml = False
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model, self.optimizer, self.scheduler, self.criterion, self.train_data, self.test_data, self.train_loader,\
            self.test_loader, self.data, self.datasets = None, None, None, None, None, None, None, None, None, None
        self.configure_dataset()
        self.configure_model(len(self.train_loader))
        self.train()

    def configure_model(self, loader_length):
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = 3
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
        self.model = nn.DataParallel(self.model)
        if self.nsml:
            bind_model(self.model)
        # TODO: add n_labels
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 1e-4  # 10^-4 good at mixup paper
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['learning_rate'], eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=loader_length * self.args['epoch'] // 10,
                                                         num_training_steps=loader_length * self.args['epoch'])
        self.criterion = nn.CrossEntropyLoss()

    def configure_dataset(self):
        def tokenize(example):
            return self.tokenizer(example['text'], padding='max_length', max_length=256, truncation=True)
        self.data = load_dataset('sentiment140')
        self.datasets = self.data.map(tokenize, batched=True, load_from_cache_file=True)
        self.datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'sentiment'])
        self.train_data = TwitterDataset(self.datasets['train'])
        self.test_data = TwitterDataset(self.datasets['test'])
        self.train_loader = Data.DataLoader(dataset=self.train_data, batch_size=32)
        self.test_loader = Data.DataLoader(dataset=self.test_data, batch_size=32)

    def train(self):
        best_acc = 0
        self.model.train()
        for epoch in self.args['epoch']:
            for batch_idx, data in enumerate(self.train_loader):
                self.step += 1
                inputs, targets = data
                outputs = self.model(**inputs)
                loss = self.criterion(outputs, targets)
                print(f"batch: {batch_idx}, loss: {loss}")
                if self.nsml:
                    nsml.report(step=self.step, scope=locals(), summary=True, train__epoch=epoch, train__loss=loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            correct = 0
            total_sample = 0
            for data in self.test_loader:
                inputs, targets = data
                outputs = self.model(**inputs)
                loss = self.criterion(outputs, targets)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == targets).sum()
                loss_total += loss.item() * len(inputs['input_ids'])
                total_sample += inputs['input_ids'].shape[0]

            acc_total = float(correct) / total_sample
            loss_total = float(loss_total) / total_sample
            if self.nsml:
                nsml.report(step=epoch, scope=locals(), summary=True, test__loss=loss_total, test_acc=acc_total)
            print(f"epoch: {epoch}, test loss: {loss_total}, test acc: {acc_total}")
            if acc_total > best_acc:
                print("Saving checkpoint")
                if self.nsml:
                    nsml.save(checkpoint=f"twitter-best-{epoch}.pt")
                else:
                    torch.save(self.model.state_dict(), f"twitter-best-{epoch}.pt")
                print("saved")

aa = Trainer()
"""
Average Sentence Length
Tone - (능동,수동 / 긍정,부정)
Ungrammatical Sentence Ratio
"""