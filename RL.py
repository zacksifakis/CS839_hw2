import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT, GPTConfig
from torch.distributions import Categorical
import re
import time
import numpy as np
import copy
import os
import pickle

batch_size = 12
block_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT()



# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# Define the RewardModel
class RewardModel(nn.Module):
    def __init__(self, gpt_model):
        super().__init__()
        self.gpt_model = gpt_model
        self.linear = nn.Linear(self.gpt_model.config.n_embd, 1)

    def forward(self, x):
        hidden_states, _ = self.gpt_model(x)
        # Use the last hidden state for prediction
        last_hidden_state = hidden_states[:, -1, :]
        reward_logits = self.linear(last_hidden_state)
        return reward_logits


def give_feedback(text):
    sentences = re.split(r'[.!?]', text)
    rewards = [1 if len(sentence.split()) <= 10 else -1 for sentence in sentences if sentence]
    return rewards if rewards else [-1]

def train_reward_model(reward_model, dataloader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = reward_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {epoch_loss / len(dataloader)}")

def rlhf_training_loop(reward_model,rlhf_model, actor_optimizer, eval_interval_rlhf, max_iters_rlhf, max_new_tokens):
    for iter in range(max_iters_rlhf):
    # Generate text
        generated_indices, log_probs, _ = rlhf_model.generate(torch.LongTensor([[enc.encode('\n')]]).to(device), max_new_tokens, block_size)

        # Predict rewards using the trained RewardModel
        predicted_rewards = reward_model(generated_indices).squeeze(-1)

        # Calculate the advantage (predicted rewards - baseline)
        baseline = predicted_rewards.mean()
        advantages = predicted_rewards - baseline

        # Calculate the policy gradient loss
        policy_gradient_loss = -torch.mean(advantages * log_probs)

        # Update the model parameters
        actor_optimizer.zero_grad()
        policy_gradient_loss.backward()
        actor_optimizer.step()

        # Record and print statistics
        if iter % eval_interval_rlhf == 0:
            print(f"Iteration {iter}: Policy Gradient Loss = {policy_gradient_loss.item()}")

def generate_text(rlhf_model, start_token, max_new_tokens, block_size):
    context = torch.tensor([[start_token]], dtype=torch.long, device=device)
    generated_indices, _, _ = rlhf_model.generate(context, max_new_tokens, block_size)
    generated_text = enc.decode(generated_indices[0].tolist())
    return generated_text

# Create a simple dataset and dataloader
class TextRewardDataset(Dataset):
    def __init__(self, encoded_samples, reward_tensors):
        self.encoded_samples = encoded_samples
        self.reward_tensors = reward_tensors

    def __len__(self):
        return len(self.encoded_samples)

    def __getitem__(self, idx):
        return self.encoded_samples[idx], self.reward_tensors[idx]

class RLHF(nn.Module):
    def __init__(self, model, reward_model):
        super().__init__()
        self.model = model
        self.reward_model = reward_model

    def forward(self, idx, targets=None):
        return self.model(idx, targets)
     
    def generate(self, idx, max_new_tokens, block_size, ref_model=None):
        # idx is (B, T) array of indices in the current context
        log_probs = torch.tensor([]).to(device)
        log_probs_ref = torch.tensor([]).to(device)
        
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # logits define instance of Iategorical class
            m = Categorical(logits=logits)
            
            # sample from the distribution
            idx_next = m.sample() # (B,)
            
            # compute the log probability of the selected tokens
            log_prob = m.log_prob(idx_next) # (B,)
            log_probs = torch.cat((log_probs, log_prob.unsqueeze(-1)), dim=-1)
            
            # compute the log probability of the selected tokens using the reference model
            if ref_model is not None:
                with torch.no_grad():
                    logits_ref, _ = ref_model(idx_cond)
                    logits_ref = logits_ref[:, -1, :]
                    probs_ref = torch.softmax(logits_ref, dim=-1)
                    log_prob_ref = torch.log(probs_ref.gather(1, idx_next.unsqueeze(-1)).squeeze(-1))
                    log_probs_ref = torch.cat((log_probs_ref, log_prob_ref.unsqueeze(-1)), dim=-1)
            
            # update the context
            idx = torch.cat((idx, idx_next.unsqueeze(-1)), dim=-1)
        
        return idx, log_probs, log_probs_ref
    


def main():
    pkl_filename = 'data/shakespeare_char/meta.pkl'
    with open(pkl_filename, 'rb') as f:
            meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    enc = lambda s: [stoi[c] for c in s]
    dec = lambda l: ''.join([itos[i] for i in l])

    # Initialize the RewardModel and train it
    reward_model = RewardModel(model)
    reward_model.to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    with open("generated_samples.txt", "r") as file:
        text_samples = file.readlines()
    rewards = [give_feedback(text) for text in text_samples]

    # Convert text samples and rewards into tensors
    encoded_samples = [torch.tensor(enc.encode(text.strip()), dtype=torch.long) for text in text_samples]
    reward_tensors = [torch.tensor([np.mean(reward)], dtype=torch.float) for reward in rewards]

    dataset = TextRewardDataset(encoded_samples, reward_tensors)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the RewardModel
    train_reward_model(reward_model, dataloader, optimizer, criterion, num_epochs=10)

    # Initialize RLHF model and optimizer
    rlhf_model = RLHF(model, reward_model)
    rlhf_model.to(device)
    actor_optimizer = torch.optim.AdamW(rlhf_model.parameters(), lr=1e-3)

    # Train and test splits
    data = torch.tensor(enc.encode("generated_samples.txt"), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    X,_ = get_batch('train') # fetch the very first batch
    X = torch.ones((X.shape[0], 1), dtype=torch.long).to(device) # for now there is no prompt
    X = X*enc.encode('\n')[0] # start with ''The'
    t0  = time.time()
    max_new_tokens = block_size
    rews_all = []
    actor_loss_all = []
    ref_coef = 0.2
    e_coef = 0.1

    eval_interval_rlhf = 2
    max_iters_rlhf = 10

    # RLHF training loop
    rlhf_training_loop(reward_model,rlhf_model, actor_optimizer, eval_interval_rlhf, max_iters_rlhf, max_new_tokens=block_size)

    # Generate text after RLHF training
    start_token_encoded = enc.encode('\n')
    generated_text = generate_text(rlhf_model, start_token_encoded, max_new_tokens=200, block_size=block_size)
    print("Generated text after RLHF training:")
    print(generated_text)

if __name__ == '__main__':
    main()
