import argparse
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym
# Import gym_donkeycar so its environments (e.g. donkey-generated-track-v0)
# get registered with Gymnasium. Without this import gym.make() will fail to
# locate the environment.
import gym_donkeycar  # noqa: F401  (unused import needed for registration)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import ViTModel, ViTImageProcessor


class DonkeyDataset(Dataset):
    """Simple dataset storing sequences of observations and actions."""

    def __init__(self, sequences: List[Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        frames, actions, next_action = self.sequences[idx]
        return torch.stack(frames), torch.stack(actions), next_action


class DonkeyTransformer(nn.Module):
    def __init__(self, hidden_size: int = 768, seq_len: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.token_proj = nn.Linear(self.vit.config.hidden_size + 2, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, images: torch.Tensor, actions: torch.Tensor):
        batch_size, seq_len, c, h, w = images.shape
        images = images.reshape(batch_size * seq_len, c, h, w)
        inputs = self.processor(images=images, return_tensors="pt")
        # HuggingFace processors always return CPU tensors. Move them to the
        # same device as the model to avoid type or device mismatches when
        # calling the ViT model.
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = self.vit(**inputs).last_hidden_state[:, 0]
        features = features.reshape(batch_size, seq_len, -1)
        tokens = torch.cat([features, actions], dim=-1)
        tokens = self.token_proj(tokens)
        tokens = tokens.permute(1, 0, 2)
        out = self.transformer(tokens)
        out = out[-1]
        return self.head(out)


def collect_data(env: gym.Env, episodes: int, seq_len: int) -> List[Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]]:
    dataset = []
    for _ in range(episodes):
        obs, _ = env.reset()
        frame_buffer: Deque[torch.Tensor] = deque(maxlen=seq_len)
        action_buffer: Deque[torch.Tensor] = deque(maxlen=seq_len)
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            frame_buffer.append(torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1))
            action_buffer.append(torch.tensor(action, dtype=torch.float32))
            if len(frame_buffer) == seq_len:
                dataset.append((list(frame_buffer), list(action_buffer), torch.tensor(action, dtype=torch.float32)))
            obs = next_obs
            done = terminated or truncated
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transformer on DonkeyCar")
    parser.add_argument("--env", default="donkey-generated-track-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    env = gym.make(args.env)
    data = collect_data(env, args.episodes, args.seq_len)
    dataset = DonkeyDataset(data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DonkeyTransformer(seq_len=args.seq_len).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        for frames, acts, next_action in loader:
            images = frames.to(device)
            actions = acts.to(device)
            pred = model(images, actions)
            loss = loss_fn(pred, next_action.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "donkey_transformer.pt")


if __name__ == "__main__":
    main()
