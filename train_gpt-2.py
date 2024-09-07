import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        assert config.n_embd % config.n_head == 0
        #create key query value for all heads,but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        
    def forward(self, x):
        B, T, C = x.size()  #Batch size, sequence length, embedding dimension
        qkv = self.c_attn(x)
        #spilt it to q,k,v
        q, k, v = torch.split(qkv, self.n_embd, dim=2)
        
        #reshape the q,k,v to accelerate compute efficiency
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, C//nh)  or use permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        #calculate the attention score and assemble it
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  #(B, nh, T, C//nh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(x)
        return y
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = "tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        
        return x 


class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   #Note: put layernorm in residual block instead out of residual block
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024   #max sequence lengths
    vocab_size: int = 50257  #number of tokens
    n_layer: int = 12        #number of layers
    n_head: int = 12         #number of heads
    n_embd: int = 768        #embedding dimension
    
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** (-0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T < self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        #forward the token and position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   #size of(T,)
        pos_embd = self.transformer.wpe(pos)  #position embeddings of shape(T, n_embd)
        tok_embd = self.transformer.wte(idx)  #position embeddings of shape(B, T, n_embd)
        x = pos_embd + tok_embd
        #forward the block of the transformer
        for block in self.transformer.h:
            x = block(x)
        logits = self.transformer.ln_f(x)  
        logits = self.lm_head(logits)  #get the score of the every vocab, the shape is(B, T, vocab_size) 
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    
    @classmethod  #duplicate the trained weight into new model
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model 


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open("input.txt","r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text))
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        
        self.CurrentIdx = 0
        
    def next_batch(self):
        B = self.B
        T = self.T
        
        buf = self.tokens[self.CurrentIdx : self.CurrentIdx + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.CurrentIdx += B*T
        return x,y
    
    
#------------------------------------------------------------------------------------------------------------------------------
# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available == True:
    device = "cuda"
print(f"using device: {device}")

# get a data batch
B, T = 4, 32
# enc = tiktoken.get_encoding('gpt2')
# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to(device)    #put the data into cpu or gpu like other data, reduce the communicate of different data
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

data_loader = DataLoaderLite(B, T)
model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

for i in range(50):
    x, y = data_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")
#------------------------------------------------------------------------------------------------------------------------------
# num_return_sequence = 5
# max_length = 30

# model = GPT.from_pretrained('gpt2')
# model.eval()

# #prefix tokens
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)  #(8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)  #(5,8)
# x = tokens

# # generate right now x is (B, T) where B=5, T=8
# # set the seed to 42
# torch.manual_seed(42)
# while x.size(1) < max_length:
#     #forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x)  #(B, T, vocab_size)
#         # take the logits at the last position
#         logits = logits[:, -1, :] #(B, vocab_size)
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # top-k_probs here becomes (5,50), tok-k_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from the top-k probabilities
#         ix = torch.multinomial(topk_probs, 1) #(B, 1)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix)  #(B,1)
#         #append to the sequence
#         x = torch.cat((x, xcol), dim=1)
        
# # print the generated text
# for i in range(num_return_sequence):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
#-------------------------------------------------------------------------------------------------------------------------