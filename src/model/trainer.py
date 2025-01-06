"""
Módulo para treinar e salvar o modelo CNN.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class ModelTrainer:
    """
    Classe responsável pelo treinamento, avaliação e salvamento do modelo.
    """

    def __init__(self, model: nn.Module, num_epochs: int, learning_rate: float):
        """
        Inicializa o objeto ModelTrainer.
        :param model: O modelo a ser treinado.
        :param num_epochs: Número de épocas para treinar.
        :param learning_rate: Taxa de aprendizado do otimizador.
        """
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()  # Função de perda para classificação
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, data_loader: DataLoader):
        """
        Método responsável por treinar o modelo.
        :param data_loader: O DataLoader contendo os dados de treinamento.
        """
        self.model.train()  # Coloca o modelo em modo de treinamento

        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for i, (spectograms, labels) in enumerate(data_loader):
                # Adicionar dimensão extra para os labels
                labels = labels.unsqueeze(1).float()

                # Zerar gradientes do otimizador
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(spectograms)
                loss = self.criterion(outputs, labels)

                # Backward pass e atualização de pesos
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 10 == 9:  # Log a cada 10 minibatches
                    logging.info(f"Época {epoch + 1}, Lote {i + 1}: Perda média = {running_loss / 10:.4f}")
                    running_loss = 0.0

        logging.info("Treinamento finalizado")

    def save_model(self, file_path: str):
        """
        Salva o modelo treinado em um arquivo.
        :param file_path: O caminho para salvar o modelo.
        """
        torch.save(self.model.state_dict(), file_path)
        logging.info(f"Modelo salvo em {file_path}")

    def evaluate(self, data_loader: DataLoader):
        """
        Avalia o modelo com os dados fornecidos.
        :param data_loader: DataLoader contendo os dados de avaliação.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for spectograms, labels in data_loader:
                outputs = self.model(spectograms)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logging.info(f"Acurácia do modelo: {accuracy:.2f}%")
        return accuracy
