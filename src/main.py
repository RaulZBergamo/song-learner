"""
Módulo principal do projeto, responsável por carregar os dados,
treinar o modelo de CNN e salvar o modelo treinado.
"""

import logging
import os

import dotenv
import torch
from torch.utils.data import DataLoader
from midi.midi_converter import MidiConverter
from data.data_set import DataSet
from data.spectogram_dataset import SpectrogramDataset
from controller.wav_controller import WavController
from utils import convert_notes_to_labels
from model.cnn import SpectrogramCNN
from model.trainer import ModelTrainer

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

midi_converter = MidiConverter()

# Hiperparâmetros
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 120

def main():
    """
    Função principal que executa o pipeline de treinamento do modelo.
    """
    data_loader = get_dataset(train=True, save_path='assets/train_spectrograms/')
    model_path = train_model(data_loader)

    data_loader = get_dataset(train=False, save_path='assets/test_spectrograms/')
    evaluate_model(model_path, data_loader)

def get_dataset(train: bool, save_path: str = None) -> DataLoader:
    """
    Função que carrega o dataset e retorna um DataLoader.

    :param train: Se True, carrega o dataset de treinamento, caso contrário, o de dataset de teste.
    :return: O DataLoader contendo os dados de treinamento.
    """
    env_model = 'TRAIN_DATASET_URL' if train else 'TESTE_DATASET_URL'

    data_set = DataSet(
        os.getenv(env_model)
    )

    data_set.download_data_set()
    wav_files = data_set.get_wav_files()

    spectograms, labels = WavController(wav_files, midi_converter, save_path).get_data()
    labels = convert_notes_to_labels(labels)

    dataset = SpectrogramDataset(spectograms, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def train_model(data_loader: DataLoader) -> str:
    """
    Função que treina o modelo de CNN e salva o modelo treinado.

    :param data_loader: O DataLoader contendo os dados de treinamento.
    :return: O caminho para o modelo treinado.
    """
    if os.path.exists('trained_cnn_model.pth'):
        logging.info("Modelo treinado já existe, pulando treinamento.")
        return 'trained_cnn_model.pth'

    logging.info("Iniciando treinamento do modelo CNN...")

    # Inicializar o modelo CNN
    model = SpectrogramCNN(num_classes=num_classes)

    # Inicializar o objeto ModelTrainer e treinar o modelo
    trainer = ModelTrainer(
        model=model,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    trainer.train(data_loader)

    # Salvar o modelo treinado
    trainer.save_model("trained_cnn_model.pth")

    logging.info("Treinamento concluído e modelo salvo com sucesso.")
    return 'trained_cnn_model.pth'

def evaluate_model(model_path: str, data_loader: DataLoader) -> None:
    """
    Função que avalia o modelo treinado.

    :param model_path: O caminho para o modelo treinado.
    :param data_loader: O DataLoader contendo os dados de teste.
    """
    logging.info("Iniciando avaliação do modelo CNN...")

    model = SpectrogramCNN(num_classes=num_classes)

    # Carregar o modelo treinado
    model.load_state_dict(torch.load(model_path))

    # Inicializar o objeto ModelTrainer e avaliar o modelo
    trainer = ModelTrainer(
        model=model,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    trainer.evaluate(data_loader)

if __name__ == '__main__':
    main()
