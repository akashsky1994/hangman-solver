
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LSTMLetterPredictor
from dataset import load_dataset, encode_word
from torch.autograd import Variable

class Trainer:
    '''
    op_type enum:[train, load_and_train, infer]
    '''
    def __init__(self, **kwargs) -> None:

        infer = kwargs.pop('infer',False)
        self.batch_size = kwargs.pop('batch_size',1)
        self.num_workers = kwargs.pop('num_workers',2)
        device = kwargs.pop('device',None)
        checkpoint_dir = kwargs.pop('checkpoint_dir',None)
        self.lr = kwargs.pop('lr',0.001)
        self.EPOCHS = kwargs.pop('epochs',5)


        self.total_chances = 6
        self.best_accuracy = -1
        self.set_device(device)
        self.totalTrainableParams = 0
        self.trainableParameters = []
        

        if not infer:
            self.model = LSTMLetterPredictor().to(self.device)
            self.trainableParameters += list(self.model.parameters())
            self.totalTrainableParams += sum(p.numel() for p in self.model.parameters() if p.requires_grad)    
            self.setup_optimizer_losses()
            self.load_dataset()
            if checkpoint_dir is not None:
                self.load_checkpoint(checkpoint_dir)
        else:
            model_path = os.path.join(checkpoint_dir,"model.pth")
            self.model = torch.load(model_path)

    def set_device(self,provided_device):
        self.n_gpus = 1
        if provided_device:
            self.device = provided_device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda' 
                self.n_gpus = torch.cuda.device_count()
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

    def load_dataset(self):
        train_dataset,dev_dataset,test_dataset,collate_fn = load_dataset()
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers,collate_fn=collate_fn)
        self.dev_loader = DataLoader(dev_dataset,batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers,collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset,batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers,collate_fn=collate_fn)
    
    def setup_optimizer_losses(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.optimizer = torch.optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4)
        self.optimizer = torch.optim.Adam(self.trainableParameters, lr=self.lr, betas=(0.9,0.999), weight_decay=0.00005)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.EPOCHS, eta_min=1e-4)
      

    def train(self):
        try:
            print("Total Trainable Parameters : {}".format(self.totalTrainableParams))    
            for epoch in tqdm(range(self.EPOCHS)):
                print("Running Epoch : {}".format(epoch))
                self.train_epoch()
                metrics = self.evaluate(self.dev_loader, "dev")
                self.scheduler.step()
                self.save_checkpoint(metrics)
                print('*' * 89)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        print("Testing with unseen data/ test data")
        unseen_metrics = self.evaluate(self.test_loader, "test")
        print(unseen_metrics)

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        total = 0
        wins = 0
        for obscured_tensor,word_feature_tensor,word_tensor in self.train_loader:
            obscured_tensor,word_feature_tensor, word_tensor = obscured_tensor.to(self.device),word_feature_tensor.to(self.device),word_tensor.to(self.device)
            self.optimizer.zero_grad()
            loss, win_flag = self.play_game(obscured_tensor,word_feature_tensor, word_tensor)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            total += word_tensor.size(0)
            wins += int(win_flag)
        
        metrics = {
            "loss": round(train_loss/total, 3),
            "accuracy": round(wins/total, 3),
        }
        print("Training Metrics :{}".format(metrics))
        return metrics
       


    def play_game(self,obscured_tensor,word_feature_tensor, word_tensor):
        batch_size = word_tensor.shape[0]
        guessed_tensor = torch.zeros(batch_size,26).to(self.device)
        chances_left = torch.full((batch_size,1), self.total_chances)
        wins = 0
        # obscured_tensor.requires_grad = True # for calculating loss
        while chances_left.sum()>0:
            out,sigmoid_out = self.model(obscured_tensor,guessed_tensor)
            guess = torch.argmax(sigmoid_out, dim=1)
            
            # Update Guessed Letters
            guess_idx = torch.LongTensor(list(enumerate(guess)))
            guessed_tensor.index_put_(indices=tuple(guess_idx.t()),values=torch.tensor(1,dtype=torch.float32)) # equivalent to : guessed_tensor[guess] = 1

            # update obscured tensor and 
            for i in range(batch_size):
                if word_tensor[i][guess[i]] and chances_left[i] and not guessed_tensor[i][guess[i]]: # could be infinite loop TODO:
                    for k,ch in enumerate(word_feature_tensor[i,:,guess[i]]):
                        if ch:
                            obscured_tensor[i][k] = sigmoid_out[i]

                    if 1 not in obscured_tensor[i,:,26]: # check if any mystery letter left
                        wins += 1
                        chances_left[i]=0
                else:
                    chances_left[i] = max(chances_left[i]-1,0)
            
        # obscured_tensor
        
        loss = self.criterion(obscured_tensor,word_feature_tensor)
        return loss, wins

    def evaluate(self,data_loader, evaluation_type="dev"):
        self.model.eval()
        test_loss = 0
        total = 0
        wins = 0
        data_loader = self.dev_loader
        if evaluation_type == "test":
            data_loader = self.test_loader
        with torch.no_grad():
            for obscured_tensor,word_feature_tensor,word_tensor in data_loader:
                obscured_tensor,word_feature_tensor, word_tensor = obscured_tensor.to(self.device),word_feature_tensor.to(self.device),word_tensor.to(self.device)
                loss,win_flag = self.play_game(obscured_tensor,word_feature_tensor,word_tensor)
                test_loss += loss.item()
                total += word_tensor.size(0)
                wins += int(win_flag)

       
        metrics =  {
            "loss": round(test_loss/total,3),
            "accuracy": round(wins/total, 3),
        }
        print("{} Metrics : {}".format(evaluation_type.upper(), metrics))
        return metrics

    def save_checkpoint(self, metrics):
        # try:
        if metrics['accuracy'] >= self.best_accuracy:
            directory_path = os.path.join("./checkpoints", "model_{}_{}".format(metrics['loss'],metrics['accuracy']))
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)


            model_weight_path = os.path.join(directory_path, "model_state_dict.pth")
            model_path = os.path.join(directory_path, "model.pth")
            optimizer_path = os.path.join(directory_path, "{}".format(type(self.optimizer).__name__))
            
            print('Saving..')
            print("Saved Model - Metrics",metrics)

            torch.save(self.model, model_path)
            torch.save(self.model.state_dict(), model_weight_path) # Saved both state_dict and whole model one for checkpointing and other for model loading during inference
            torch.save(self.optimizer.state_dict(), optimizer_path)
            
            self.best_accuracy = metrics['accuracy']
            self.best_loss = metrics['loss']
        # except Exception as e:
        #     print("Error:",e)
            
    def load_checkpoint(self, checkpoint_dir):
        try:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            print(checkpoint_dir)
            assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
            
            model_path = os.path.join(checkpoint_dir,"model_state_dict.pth")
            self.model.load_state_dict(torch.load(model_path))
            optim_state_dict = torch.load(os.path.join(checkpoint_dir, "{}".format(type(self.optimizer).__name__)))
            self.optimizer.load_state_dict(optim_state_dict)
        except Exception as e:
            print(e)

    def infer(self, obscured_word, guessed_letters):
        guessed_letters_tensor = np.zeros(26, dtype=np.float32)
        for i in guessed_letters:
            guessed_letters_tensor[ord(i)-97] = 1
        obscured_word_tensor = encode_word(obscured_word)
        out,sigmoid_out = self.model(obscured_word_tensor, guessed_letters_tensor)
        guess = torch.argmax(out, dim=2).item()
        guess = chr(guess + 97)
        return guess


if __name__ == "__main__":
    trainer = Trainer(batch_size=64,epochs=10)
    trainer.train()  

    
