"""
Módulo para carregar os dados de espectrogramas e notas musicais em um dataset.
"""

from typing import List

import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    """
    Classe para carregar os dados de espectrogramas e notas musicais em um dataset.
    """
    def __init__(self) -> None:
        """
        Inicializa o dataset com espectrogramas e rótulos numéricos.
        
        :param spectrograms: Lista de arrays contendo os espectrogramas.
        """
        self.spectrograms = []
        self.labels = []

    def __len__(self) -> int:
        """
        Retorna o número total de amostras no dataset.
        :return: Número de amostras.
        """
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retorna o espectrograma e o rótulo na posição idx.
        
        :param idx: Índice da amostra.
        :return: Espectrograma (tensor) e rótulo (tensor) correspondentes.
        """
        # Convertendo espectrograma em tensor e adicionando o canal para CNN
        spectrogram = torch.tensor(self.spectrograms[idx], dtype=torch.float32).unsqueeze(0)
        # Convertendo o rótulo em tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spectrogram, label
    
    def add_sample(self, spectrogram: torch.Tensor, label: torch.Tensor) -> None:
        """
        Adiciona um espectrograma e rótulo ao dataset.
        
        :param spectrogram: Espectrograma a ser adicionado.
        :param label: Rótulo correspondente ao espectrograma.
        """
        self.spectrograms.append(spectrogram)
        self.labels.append(label)
