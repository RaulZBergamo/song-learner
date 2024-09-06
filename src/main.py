"""
Módulo principal do projeto.
"""

import logging
import os

import dotenv
from midi.midi_converter import MidiConverter
from data.data_set import DataSet

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

    data_set.get_data_set()


if __name__ == '__main__':
    main()
