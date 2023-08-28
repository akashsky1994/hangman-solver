import torch
from torch.autograd import Variable

class LSTMLetterPredictor(torch.nn.Module):
    def __init__(self, hidden_size = 32, target_size=26, dropout=0.2, num_layers = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = torch.nn.ModuleList() 
        input_size = 27
        for _ in range(num_layers):
            self.lstm_cells.append(torch.nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        # self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size + 26, target_size) #because of adding [1x26] size guessed tensor to the output

    def init_state(self, batch_size, device):
        cx = [torch.zeros((batch_size, self.hidden_size)).to(device)]*self.num_layers
        hx = [torch.zeros((batch_size, self.hidden_size)).to(device)]*self.num_layers
        for i in range(self.num_layers):
            # Weights initialization
            torch.nn.init.xavier_normal_(hx[i])
            torch.nn.init.xavier_normal_(cx[i])
        return hx,cx
    

    def forward(self,obscured_tensor,guessed_tensor, aggregation_type="avg"):
        word_seq_length = obscured_tensor.shape[1]
        batch_size = obscured_tensor.shape[0]
        hx,cx = self.init_state(batch_size,obscured_tensor.device)
        output_sequence = []
        obscured_tensor = obscured_tensor.view(word_seq_length,batch_size,-1)
        
        for i in range(word_seq_length):
            out = obscured_tensor[i]
            for j,lstm_cell in enumerate(self.lstm_cells):
                hx[j], cx[j] = lstm_cell(out, (hx[j], cx[j]))
                out = hx[j]

            out = torch.cat([out,guessed_tensor], 1)
            output_sequence.append(out)
        
        output_sequence = torch.stack(output_sequence)

        if aggregation_type=="sum":
            output_sequence = torch.sum(output_sequence, dim=0)
        elif aggregation_type=="avg":
            output_sequence = torch.mean(output_sequence, dim=0)
        elif aggregation_type=="last":
            output_sequence = output_sequence[-1]
        else:
            raise NotImplementedError("Aggregation Type {} not implement".format(aggregation_type))
        #TODO: trying putting linear after the aggregation

        output_sequence = self.fc(output_sequence)

        return output_sequence, torch.sigmoid(output_sequence)