"""
Módulo para criar um modelo de CNN avançada para classificação de espectrogramas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):
    """
    Modelo de Rede Neural Convolucional projetado para classificação de espectrogramas de áudio.
    Inclui múltiplas camadas convolucionais seguidas por pooling, dropout e fully connected layers.
    """
    def __init__(self) -> None:
        """
        Instancia o modelo SpectrogramCNN.
        
        :param num_classes: Número de classes para a classificação final.
        """
        super(SpectrogramCNN, self).__init__()
        
        # Primeira camada convolucional (entrada: 1 canal, saída: 32 filtros)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normalização em batch após a primeira convolução
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling para reduzir as dimensões
        
        # Segunda camada convolucional (entrada: 32 filtros, saída: 64 filtros)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Normalização em batch
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Terceira camada convolucional (entrada: 64 filtros, saída: 128 filtros)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Normalização em batch
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(0.5)
        
        # Camada totalmente conectada (fully connected)
        self.fc1 = nn.Linear(128 * 128 * 15, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Define a passagem direta do modelo.
        
        :param spectrogram: Tensor de entrada (espectrograma).
        :return: Saída do modelo.
        """
        # Passagem pelas camadas convolucionais com Pooling e BatchNorm
        spectrogram = self.pool1(self.bn1(self.conv1(spectrogram)))
        spectrogram = self.pool2(self.bn2(self.conv2(spectrogram)))
        spectrogram = self.pool3(self.bn3(self.conv3(spectrogram)))

        # Flatten para as fully connected layers
        spectrogram = spectrogram.view(spectrogram.size(0), -1)
        
        # Passagem pelas fully connected layers com dropout
        spectrogram = self.fc1(spectrogram)
        spectrogram = self.dropout(spectrogram)
        spectrogram = self.fc2(spectrogram)
        
        return spectrogram
