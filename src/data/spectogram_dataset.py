"""
Módulo para carregar os dados de espectrogramas e notas musicais em um dataset.
"""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    """
    Classe para carregar os dados de espectrogramas e notas musicais em um dataset.
    """
    def __init__(self, spectrograms: List[torch.Tensor], labels: List[int]):
        """
        Inicializa o dataset com espectrogramas e rótulos numéricos.
        
        :param spectrograms: Lista de arrays contendo os espectrogramas.
        :param labels: Lista de rótulos numéricos correspondentes às notas musicais.
        """
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self) -> int:
        """
        Retorna o número total de amostras no dataset.
        :return: Número de amostras.
        """
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
