import torch
import torch.nn as nn

class LSTMmodule(torch.nn.Module):
    """
    This is the LSTM implementation used for linking group
    features coming from different segments.
    """
    def __init__(self, img_feature_dim, num_frames, num_class, num_layers = 1):
        super(LSTMmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = 512
        self.hidden_dim = 1024
        self.num_layers = num_layers
        self.classifier = nn.Sequential(
                                       nn.LSTM(self.num_bottleneck, self.hidden_dim, self.num_layers),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.num_class),
                                       )
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = nn.Linear(self.num_frames * self.img_feature_dim,
                                                self.num_bottleneck * self.num_frames)(input)
        input = input.view(input.size(0), self.num_frames, self.num_bottleneck)
        input = self.classifier(input)
        return input


def return_LSTM(relation_type, img_feature_dim, num_frames, num_class):
    LSTM = LSTMmodule(img_feature_dim, num_frames, num_class)
    return LSTM
