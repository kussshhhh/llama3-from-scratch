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

q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
print(q_per_token_split_into_pairs.shape)

# [8, 32, 2]

# here we split the query vectors into pairs we pairs and then we apply rotational angle shift to each pair

# we now have a vector of size[8 x 32 x 2]

zero_to_one_split_into_32_parts = torch.tensor(range(32))/32
print(zero_to_one_split_into_32_parts)

freqs = 1.0 / (rope_theta **  zero_to_one_split_into_32_parts)
print(freqs)

freqs_for_each_token = torch.outer(torch.arange(8), freqs)
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
print(freqs_cis.shape)

#veiwing the third row of freqs_cis
value = freqs_cis[3]
plt.figure()
for i, element in enumerate(value[:8]):
    plt.plot([0,element.real], [0, element.imag], color = 'blue', linewidth=1, label=f"Index: {i}")
    plt.annotate(f"{i}", xy=(element.real, element.imag), color='red')

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Plot of one row of freq_cis')
plt.show()

# converting q vector to complex numbers and rotating them
q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
print(q_per_token_as_complex_numbers.shape)

q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis
print(q_per_token_as_complex_numbers_rotated.shape)

# converting back to real numbers
q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated)
print(q_per_token_split_into_pairs_rotated.shape)

# merging the pairs back
q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
print(q_per_token_rotated.shape)

# load the key weights and do the same shit
k_layer0 = model["layers.0.attention.wk.weight"]
k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim)
print(k_layer0.shape)

k_layer0_head0 = k_layer0[0]
print(k_layer0_head0.shape)

k_per_token = torch.matmul(token_embeddings, k_layer0_head0.T)
print(k_per_token.shape)

# rotate the key vectors too
k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
print(k_per_token_rotated.shape)

# calculate attention scores
qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (head_dim ** 0.5)
print(qk_per_token.shape)

# mask future tokens (causal masking)
mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
mask = torch.triu(mask, diagonal=1)
qk_per_token_after_masking = qk_per_token + mask

# apply softmax to get attention probabilities  
qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
print(qk_per_token_after_masking_after_softmax.shape)

# load value weights
v_layer0 = model["layers.0.attention.wv.weight"]
v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim)
print(v_layer0.shape)

v_layer0_head0 = v_layer0[0]
print(v_layer0_head0.shape)

v_per_token = torch.matmul(token_embeddings, v_layer0_head0.T)
print(v_per_token.shape)

# apply attention to values
qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
print(qkv_attention.shape)

print("single head attention complete, now doing multi-head attention...")

# multi-head attention for layer 0
qkv_attention_store = []

for head in range(n_heads):
    q_layer0_head = q_layer0[head]
    k_layer0_head = k_layer0[head//4]  # key weights shared across 4 heads  
    v_layer0_head = v_layer0[head//4]  # value weights shared across 4 heads
    
    q_per_token = torch.matmul(token_embeddings, q_layer0_head.T)
    k_per_token = torch.matmul(token_embeddings, k_layer0_head.T)
    v_per_token = torch.matmul(token_embeddings, v_layer0_head.T)

    # apply RoPE to queries
    q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
    q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
    q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

    # apply RoPE to keys
    k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
    k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
    k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

    # attention calculation
    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (head_dim ** 0.5)
    mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
    mask = torch.triu(mask, diagonal=1)
    qk_per_token_after_masking = qk_per_token + mask
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
    
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention_store.append(qkv_attention)

print(f"completed attention for {len(qkv_attention_store)} heads")

# concat all attention heads
stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
print(stacked_qkv_attention.shape)

# apply output projection
w_layer0 = model["layers.0.attention.wo.weight"]
embedding_delta = torch.matmul(stacked_qkv_attention, w_layer0.T)
print(embedding_delta.shape)

# residual connection
embedding_after_edit = token_embeddings_unnormalized + embedding_delta
print(embedding_after_edit.shape)

# normalize before feed forward
embedding_after_edit_normalized = rms_norm(embedding_after_edit, model["layers.0.ffn_norm.weight"])
print(embedding_after_edit_normalized.shape)

# feed forward network (SwiGLU)
w1 = model["layers.0.feed_forward.w1.weight"]
w2 = model["layers.0.feed_forward.w2.weight"] 
w3 = model["layers.0.feed_forward.w3.weight"]

# SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T
output_after_feedforward = torch.matmul(
    torch.nn.functional.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * 
    torch.matmul(embedding_after_edit_normalized, w3.T), 
    w2.T
)
print(output_after_feedforward.shape)

# final residual connection for layer 0
layer_0_embedding = embedding_after_edit + output_after_feedforward
print(layer_0_embedding.shape)

print("layer 0 complete! now running all 16 layers...")

# complete transformer - all 16 layers
final_embedding = token_embeddings_unnormalized

for layer in range(n_layers):
    print(f"processing layer {layer}")
    
    qkv_attention_store = []
    
    # normalize input embeddings
    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
    
    # load attention weights for this layer
    q_layer = model[f"layers.{layer}.attention.wq.weight"]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
    
    k_layer = model[f"layers.{layer}.attention.wk.weight"]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
    
    v_layer = model[f"layers.{layer}.attention.wv.weight"]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
    
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    
    # multi-head attention
    for head in range(n_heads):
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head//4]  # grouped query attention
        v_layer_head = v_layer[head//4]  
        
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

        # apply RoPE 
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

        # attention calculation
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (head_dim ** 0.5)
        mask = torch.full((len(final_embedding), len(final_embedding)), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
        
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    # combine all heads
    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
    
    # output projection
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
    
    # residual connection
    embedding_after_edit = final_embedding + embedding_delta
    
    # normalize before feed forward
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
    
    # feed forward network (SwiGLU)
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
    
    output_after_feedforward = torch.matmul(
        torch.nn.functional.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * 
        torch.matmul(embedding_after_edit_normalized, w3.T), 
        w2.T
    )
    
    # final residual connection
    final_embedding = embedding_after_edit + output_after_feedforward

print("all layers complete! now doing final normalization and getting logits...")

# final normalization
final_embedding = rms_norm(final_embedding, model["norm.weight"])
print(final_embedding.shape)

# decode to logits using output weights
logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
print(logits.shape)

# get next token prediction
next_token = torch.argmax(logits, dim=-1)
print(f"predicted next token: {next_token}")
print(f"decoded token: '{tokenizer.decode([next_token.item()])}'")

print("let's fucking go! implementation complete!")

def generate_text(prompt, max_tokens=50):
    """generate text using our llama implementation"""
    tokens = [128000] + tokenizer.encode(prompt)
    
    for _ in range(max_tokens):
        tokens_tensor = torch.tensor(tokens)
        
        # embedding layer
        embedding_layer = torch.nn.Embedding(vocab_size, dim)
        embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
        token_embeddings_unnormalized = embedding_layer(tokens_tensor).to(torch.bfloat16)
        
        # get freqs for current sequence length
        zero_to_one_split_into_32_parts = torch.tensor(range(32))/32
        freqs = 1.0 / (rope_theta ** zero_to_one_split_into_32_parts)
        freqs_for_each_token = torch.outer(torch.arange(len(tokens)), freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
        
        # process through all layers
        final_embedding = token_embeddings_unnormalized
        
        for layer in range(n_layers):
            qkv_attention_store = []
            
            layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
            
            q_layer = model[f"layers.{layer}.attention.wq.weight"]
            q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
            
            k_layer = model[f"layers.{layer}.attention.wk.weight"]
            k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
            
            v_layer = model[f"layers.{layer}.attention.wv.weight"]
            v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
            
            w_layer = model[f"layers.{layer}.attention.wo.weight"]
            
            for head in range(n_heads):
                q_layer_head = q_layer[head]
                k_layer_head = k_layer[head//4]
                v_layer_head = v_layer[head//4]
                
                q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
                k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
                v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

                q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
                q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
                q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
                q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

                k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
                k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
                k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
                k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

                qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (head_dim ** 0.5)
                mask = torch.full((len(final_embedding), len(final_embedding)), float("-inf"))
                mask = torch.triu(mask, diagonal=1)
                qk_per_token_after_masking = qk_per_token + mask
                qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
                
                qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
                qkv_attention_store.append(qkv_attention)

            stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
            embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
            embedding_after_edit = final_embedding + embedding_delta
            embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
            
            w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
            
            output_after_feedforward = torch.matmul(
                torch.nn.functional.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * 
                torch.matmul(embedding_after_edit_normalized, w3.T), 
                w2.T
            )
            
            final_embedding = embedding_after_edit + output_after_feedforward

        # final output
        final_embedding = rms_norm(final_embedding, model["norm.weight"])
        logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        
        tokens.append(next_token.item())
        
        # break if end token (token 128001 is <|end_of_text|>)
        if next_token.item() == 128001:
            break
    
    return tokenizer.decode(tokens[1:])  # skip begin token

# test the generation function
print("\n=== testing text generation ===")
test_prompt = "the secret to the universe is"
generated_text = generate_text(test_prompt, max_tokens=20)
print(f"prompt: {test_prompt}")
print(f"generated: {generated_text}")


