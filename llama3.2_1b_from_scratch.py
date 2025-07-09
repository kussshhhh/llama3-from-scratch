from pathlib import Path
import torch 
import tiktoken
from tiktoken.load import load_tiktoken_bpe 
import torch 
import json
import matplotlib.pyplot as plt 
import os

tokenizer_path = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model")

special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]

mergeable_ranks = load_tiktoken_bpe(tokenizer_path)

tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

print(tokenizer.decode(tokenizer.encode("hello world!")))

model = torch.load(os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth"), map_location=torch.device('cpu'))
print(json.dumps(list(model.keys())[:20], indent=2))

with open(os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/params.json"), "r") as f:
    config = json.load(f)
print(json.dumps(config, indent=2))

dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

prompt = "the secret to the universe is "
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)

tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
print(prompt_split_as_tokens)

embedding_layer = torch.nn.Embedding(vocab_size, dim)
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
print(token_embeddings_unnormalized.shape)

def rms_norm(tensor, norm_weights):
    norms = (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) 
    print(norms.shape)
    return norms * norm_weights

token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"])
print(token_embeddings.shape)

print(
    model["layers.0.attention.wq.weight"].shape,
    model["layers.0.attention.wk.weight"].shape,
    model["layers.0.attention.wv.weight"].shape,
    model["layers.0.attention.wo.weight"].shape
)

# torch.Size([2048, 2048]) torch.Size([512, 2048]) torch.Size([512, 2048]) torch.Size([2048, 2048])

q_layer0 = model["layers.0.attention.wq.weight"]
head_dim = q_layer0.shape[0] // n_heads
q_layer0 = q_layer0.view(n_heads, head_dim, dim)
print(q_layer0.shape)
# [32, 64, 2048] // btw 32 * 64 =  2048 so we just seperated matrices for all 32 heads

# aigth now the first head of of first layer

q_layer0_head0 = q_layer0[0]
print(q_layer0_head0.shape)

# [64, 2048]

q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T) # multiplying by transpose
print(q_per_token.shape)
# [8, 2048] x [2048, 64] = [8, 64]

# and now we have q vector of the all 8 tokens of the first head 

# now we do RoPe rotary positional embeddings

