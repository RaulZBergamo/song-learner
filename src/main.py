"""
Módulo principal do projeto.
"""

import logging
import os

import dotenv
from midi.midi_converter import MidiConverter
from data.data_set import DataSet
from data.spectogram_dataset import SpectrogramDataset
from controller.wav_controller import WavController
from utils import convert_notes_to_labels

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

midi_converter = MidiConverter()

data_set = DataSet(
    os.getenv('DATASET_URL')
)

def main():
    """
    Método principal do projeto.
    """

    data_set.download_data_set()
    wav_files = data_set.get_wav_files()

    spectograms, labels = WavController(wav_files, midi_converter).get_data_for_training()

    labels = convert_notes_to_labels(labels)

    dataset = SpectrogramDataset(spectograms, labels)

    print(dataset)

if __name__ == '__main__':
    main()
