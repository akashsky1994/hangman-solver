import torch
from torch.autograd import Variable

class LSTMLetterPredictor(torch.nn.Module):
    def __init__(self,input_size = 26, hidden_size = 32, target_size=26, dropout=0.2, num_layers = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,batch_first=True,bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size*2 + 26, target_size) #because of adding [1x26] size guessed tensor to the output

    def init_state(self, batch_size, device):
        cx = torch.zeros((batch_size, self.hidden_size)).to(device)
        hx = torch.zeros((batch_size, self.hidden_size)).to(device)
        
        # Weights initialization
        torch.nn.init.xavier_normal_(hx)
        torch.nn.init.xavier_normal_(cx)
        return hx,cx
    

    def forward(self,obscured_tensor,guessed_tensor, aggregation_type="sum"):
        word_seq_length = obscured_tensor.shape[1]
        batch_size = obscured_tensor.shape[0]
        output_sequence = []

        hx, cx = self.lstm(obscured_tensor)
        for i in range(word_seq_length):
            out = torch.cat([hx[:,i,:],guessed_tensor],1)
            output_sequence.append(out)

        output_sequence = torch.stack(output_sequence)

        output_sequence = self.fc(output_sequence)
        output_sequence = output_sequence.view(batch_size, word_seq_length,-1)

        if aggregation_type=="sum":
            aggregated_output = torch.sum(output_sequence, dim=1)
        elif aggregation_type=="avg":
            aggregated_output = torch.mean(output_sequence, dim=1)
        elif aggregation_type=="last":
            aggregated_output = output_sequence[:,-1,:]
        else:
            raise NotImplementedError("Aggregation Type {} not implement".format(aggregation_type))
            
        return output_sequence,aggregated_output