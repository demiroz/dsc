import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters 
batch_size= 32 # howmany independent sequences will we process in parallel?
block_size= 8 # what is the maximum contest length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd= 32
#--------------------
torch.manual_seed(1337)


#read the data in to inspect it
with open ('/Users/SHSU/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Springboard/dsc/Capstone Three/HC_DATA/medquad_qa_pairs.csv', 'r') as f:
    df = f.read()
print('length of dataset in characters:', len(df))
print(df[:1000])
chars = sorted(list(set(df)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
stoi ={ch:i for i, ch in enumerate (chars) }
itos ={i:ch for i, ch in enumerate (chars) }
encode = lambda s: [stoi[c] for c in s] # encoder : take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decpder: take a lost of integers, output a string

print(encode("hii there"))
print(decode (encode('hii there')))
#let's now encode the entire text dataset and store it into a torch.tensor
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(df), dtype= torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characers we lloked at earlier will to the GPT look like this.
#let's now split up the data into train and validation sets
n= int(0.9 * len(data)) #first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
block_size = 8
train_data [:block_size + 1]
x= train_data[:block_size]
y= train_data[1:block_size + 1]
for t in range (block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we procss in parallel?
block_size = 8 # what is the maximum context length for predictions?


# data loading
def get_batch(split): 
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x= torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): #time dimension
        context= xb[b, :t+1]
        target = yb[b,t]
        print(f'when input is {context.tolist()} the target:{target}')
print(xb) # our input to the transformer


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model (X, Y)
            losses[k] = loss.item()
        out[split]= losses.mean()
    model.train()
    return out


# Super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next roken from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_Tabke = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size )

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb= self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        x= tok_emb + pos_emb #(B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)


        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self(idx)
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax (logits, dim=-1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples =1) # (B,1)
            #append sampled index to the running sequence 
            idx=torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

model= BigramLanguageModel()
m = model.to(device)


logits, loss = m(xb, yb)
out = m(xb, yb)
print (logits.shape)
print(loss)

idx= torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx= torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


# create a PyTorch optimizer
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
batch_size =32
for steps in range (10000):
    # sample a batch of data
    xb, yb = get_batch ('train')

    #evaluate the loss
    logits, loss= m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(idx= torch.zeros((1,1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))
