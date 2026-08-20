# train a miniature character-level shakespeare model with MLA attention
# good for debugging and playing on macbooks and such
# run: python train_mla.py config/train_shakespeare_char_mla.py --device=cpu --compile=False

out_dir = 'out-shakespeare-char-mla'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-mla'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# ---- MLA settings ----
# n_embd=384, n_head=6 -> head_dim=64, split as 48 nope + 16 rope.
kv_lora_rank = 128    # compressed K/V latent dim. KV cache per token per layer drops from
                      # 2*384=768 numbers (MHA) to 128+16=144 -> 5.33x
q_lora_rank = 0       # keep direct Q projection for this tiny model
qk_rope_head_dim = 16 # DeepSeek decoupled RoPE: k_R gets its own W_KR (one head shared by
                      # all heads) and is cached beside c_kv. The remaining 48 nope dims are
                      # absorbable, which is what the absorbed decode path needs.
                      # Set to 0 for the legacy layout (whole head rotated, no absorption).
v_head_dim = 0        # 0 = same as the q/k head dim (64)
use_rope = True       # RoPE instead of learned wpe
rope_theta = 10000.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
