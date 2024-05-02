import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    """
    Torch implementation of a neural network for supervised learning.

    The neural network for prediction is a Multilayer Perceptron defined by users.
    """

    def __init__(
            self, 
            predictor_dims=[20, 512, 1024], 
            predictor_non_linear_activation='relu',
            predictor_bias=True,
            dropout=0.0
            ):
        
        """
        Args:       
            predictor_dims (list): list of integers, where the first element is the dimension of the input, and the last element is the dimension of the output.
            predictor_non_linear_activation (str): 'relu' or 'sigmoid'. Non-linear activation function for the hidden layers.
            predictor_bias (bool): whether to include bias terms in the hidden layers.
            dropout (float): dropout rate.
        """

        super(Predictor, self).__init__()

        self.predictor_dims = predictor_dims
        self.predictor_non_linear_activation = predictor_non_linear_activation
        self.predictor_bias = predictor_bias

        self.dropout = nn.Dropout(p=dropout)
        if predictor_non_linear_activation is not None:
            self.encoder_nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[predictor_non_linear_activation]
        if predictor_non_linear_activation is not None:
            self.decoder_nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[predictor_non_linear_activation]

        self.neural_net = nn.ModuleDict({
            f'pred_{i}': nn.Linear(predictor_dims[i], predictor_dims[i+1], bias = predictor_bias)
            for i in range(len(predictor_dims)-1)
        })

    def forward(self, x, M_prediction):
        """
        Forward pass.
        """
        if M_prediction is not None:
            hid = torch.cat((x, M_prediction), dim=1)
        else:
            hid = x
        for i, (_,layer) in enumerate(self.neural_net.items()):
            hid = self.dropout(layer(hid))
            if i < len(self.neural_net)-1 and self.predictor_non_linear_activation is not None:
                hid = self.encoder_nonlin(hid)
        return hid

class PredictorMLP(nn.Module):
    """
    Torch implementation of a neural network for supervised learning.

    The neural network for prediction is a Multilayer Perceptron defined by users.
    """

    def __init__(
            self,
            predictor_dims=[20, 512, 1024],
            predictor_non_linear_activation='relu',
            predictor_bias=True,
            dropout=0.0
    ):
        super(PredictorMLP, self).__init__()

        self.predictor_dims = predictor_dims

        predictor_layers = []
        for i in range(len(predictor_dims) - 1):
            predictor_layers.append(nn.Linear(predictor_dims[i], predictor_dims[i + 1], bias=predictor_bias))
            predictor_layers.append(nn.Dropout(dropout))
            if i < len(predictor_dims) - 2:
                if predictor_non_linear_activation == 'relu':
                    predictor_layers.append(nn.ReLU())
                elif predictor_non_linear_activation == 'sigmoid':
                    predictor_layers.append(nn.Sigmoid())
                else:
                    raise ValueError(f'Unknown activation function {predictor_non_linear_activation}')
        self.predictor = nn.Sequential(*predictor_layers)

    def forward(self, x, M_prediction):
        """
        Forward pass.
        """
        if M_prediction is not None:
            x = torch.cat((x, M_prediction), dim=1)
        return self.predictor(x)
