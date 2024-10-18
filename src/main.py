"""
Módulo principal do projeto, responsável por carregar os dados,
treinar o modelo de CNN e salvar o modelo treinado.
"""

import logging
import os
import dotenv
from torch.utils.data import DataLoader

from midi.midi_converter import MidiConverter
from data.data_set import DataSet
from data.spectogram_dataset import SpectrogramDataset
from controller.wav_controller import WavController
from utils import convert_notes_to_labels
from model.cnn import SpectrogramCNN  # Importa o modelo de CNN
from model.trainer import ModelTrainer  # Importa a classe de treinamento

dotenv.load_dotenv()

# Configurações de log
logging.basicConfig(level=logging.INFO)

# Inicializa MidiConverter e DataSet
midi_converter = MidiConverter()
data_set = DataSet(
    os.getenv('DATASET_URL')
)

# Hiperparâmetros
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 120

def main():
    """
    Função principal que executa o pipeline de treinamento do modelo.
    """
    # Baixar dataset e obter arquivos WAV
    data_set.download_data_set()
    wav_files = data_set.get_wav_files()

    # Extrair espectrogramas e labels (notas musicais)
    spectograms, labels = WavController(wav_files, midi_converter).get_data_for_training()
    labels = convert_notes_to_labels(labels)

    # spectrograms: Uma lista de np.ndarrays, no total tem 4096 itens
    # Cada item possui 173 colunas e 1025 linhas

    # Criar o dataset e DataLoader
    # - dataset é uma instância de SpectrogramDataset que herda de DataSet (PyTorch)
    # com isso temos basicamente um DataFrame, onde cada linha tem arrays com as mesmas 
    # medidas do espectrograma, basicamente um array de 1 por 1025 por 173
    # - data_loader é uma instância de DataLoader (PyTorch) que é responsável por carregar
    # os dados em lotes para o treino
    dataset = SpectrogramDataset(spectograms, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

if __name__ == '__main__':
    main()
