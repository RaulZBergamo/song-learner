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
from model.cnn import SpectrogramCNN
from model.trainer import ModelTrainer
from repositories.huggingface_repository import HugginfaceRepository

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

midi_converter = MidiConverter()
wav_controller = WavController(midi_converter)

hub_repo = HugginfaceRepository(os.getenv('HUGGINGFACEHUB_USERNAME'))

# Hiperparâmetros
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 120

def main():
    """
    Função principal que executa o pipeline de treinamento do modelo.
    """
    dataset = get_dataset(train=True)
    model_path = train_model(dataset)

    dataset = get_dataset(train=False)
    evaluate_model(model_path, dataset)

def get_dataset(train: bool) -> DataSet:
    """
    Função que carrega o dataset e retorna um DataLoader.

    :param train: Se True, carrega o dataset de treinamento, caso contrário, o de dataset de teste.
    :return: O DataLoader contendo os dados de treinamento.
    """
    env_model = 'TRAIN_DATASET_URL' if train else 'TESTE_DATASET_URL'

    return DataSet(
        data_set_url=os.getenv(env_model),
        hub_repo=hub_repo,
        update_dataset=False
    ).download_data_set()

def train_model(dataset: DataSet) -> str:
    """
    Função que treina o modelo de CNN e salva o modelo treinado.

    :param data_loader: O DataLoader contendo os dados de treinamento.
    :return: O caminho para o modelo treinado.
    """
    if os.path.exists('trained_cnn_model.pth'):
        logging.info("Modelo treinado já existe, pulando treinamento.")
        return 'trained_cnn_model.pth'

    logging.info("Iniciando treinamento do modelo CNN...")

    spectograms_dataset = SpectrogramDataset()

    for item in dataset['train']:
        spectogram, label = wav_controller.load_wav(item['audio'])
        spectograms_dataset.add_sample(spectogram, label)

    data_loader = DataLoader(spectograms_dataset, batch_size=batch_size, shuffle=True)

    # Inicializar o modelo CNN
    model = SpectrogramCNN()

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
