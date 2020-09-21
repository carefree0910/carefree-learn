import torch

rnn_dict = {"LSTM": torch.nn.LSTM, "GRU": torch.nn.GRU, "RNN": torch.nn.RNN}

__all__ = ["rnn_dict"]
