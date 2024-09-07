"""
Módulo para obter os dados de áudio a serem utilizados.
"""

import logging
import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle
from midi.midi_converter import MidiConverter


class WavController:
    """
    Classe responsável por obter os dados de áudio e gerar espectrogramas e notas musicais para treinamento.
    """

    def __init__(self, file_paths: List[str], midi_converter: MidiConverter, save_path: str = None) -> None:
        """
        Instancia um novo objeto WavController.
        :param file_paths: Lista de caminhos de arquivos de áudio.
        :param midi_converter: Instância de MidiConverter para converter pitches em notas musicais.
        :param save_path: Caminho para salvar os espectrogramas e notas (opcional).
        """
        self.file_paths = file_paths
        self.midi_converter = midi_converter
        self.save_path = save_path or os.getcwd() + '/assets/spectrograms/'
        os.makedirs(self.save_path, exist_ok=True)  # Cria o diretório se não existir

        self.audio_data = []
        self.sample_rates = []
        self.spectrograms = []
        self.notes = []

    def load_wavs(self, regenerate: bool = False) -> None:
        """
        Método responsável por carregar arquivos de áudio .wav ou carregar espectrogramas e notas salvas.
        :param regenerate: Se True, regera os espectrogramas e notas musicais, mesmo que existam salvos.
        """
        for file_path in self.file_paths:
            file_name = os.path.basename(file_path)
            spectrogram_file = os.path.join(self.save_path, f"{file_name}_spectrogram.npy")
            note_file = os.path.join(self.save_path, f"{file_name}_note.pkl")

            if os.path.exists(spectrogram_file) and os.path.exists(note_file) and not regenerate:
                # Carregar espectrograma e notas musicais salvas
                self.spectrograms.append(np.load(spectrogram_file))
                with open(note_file, 'rb') as f:
                    self.notes.append(pickle.load(f))
                logging.info(f'Espectrograma e nota carregados para {file_name}.')
            else:
                # Carregar o arquivo de áudio e gerar os dados
                logging.info(f'Carregando e gerando espectrograma para {file_name}.')
                self._process_file(file_path)
                self._save_data(file_name)

    def _process_file(self, file_path: str) -> None:
        """
        Processa um único arquivo de áudio: gera espectrograma e converte o pitch para nome de nota.
        :param file_path: Caminho do arquivo .wav.
        """
        audio_data, sample_rate = librosa.load(file_path)
        self.audio_data.append(audio_data)
        self.sample_rates.append(sample_rate)

        # Gera o espectrograma
        S = np.abs(librosa.stft(audio_data))
        spectrogram = librosa.amplitude_to_db(S, ref=np.max)
        self.spectrograms.append(spectrogram)

        # Extrai o pitch do nome do arquivo e converte para o nome da nota
        pitch = self._extract_pitch_from_filename(file_path)
        note_name = self.midi_converter.midi_to_note_name(pitch)  # Converte para nome da nota musical
        self.notes.append(note_name)

    def _extract_pitch_from_filename(self, file_path: str) -> int:
        """
        Extrai o pitch do nome do arquivo .wav.
        Assumimos que o nome dos arquivos segue o padrão "nome_pitch_xxx.wav".
        :param file_path: Caminho do arquivo .wav.
        :return: Pitch extraído do nome do arquivo.
        """
        try:
            file_name = os.path.basename(file_path)
            pitch_str = file_name.split('-')[1]
            pitch = int(pitch_str)
            logging.info(f'Pitch extraído do nome do arquivo {file_name}: {pitch}')
            return pitch
        except Exception as e:
            logging.error(f'Erro ao extrair pitch do nome do arquivo {file_path}: {e}')
            raise ValueError(f'Nome do arquivo {file_path} não está no formato esperado.')

    def _save_data(self, file_name: str) -> None:
        """
        Salva o espectrograma e o nome da nota em arquivos separados.
        :param file_name: Nome do arquivo de áudio (sem caminho completo).
        """
        spectrogram_file = os.path.join(self.save_path, f"{file_name}_spectrogram.npy")
        note_file = os.path.join(self.save_path, f"{file_name}_note.pkl")

        # Salva espectrograma como arquivo .npy
        np.save(spectrogram_file, self.spectrograms[-1])

        # Salva a nota (nome da nota) usando pickle
        with open(note_file, 'wb') as f:
            pickle.dump(self.notes[-1], f)

        logging.info(f'Espectrograma e nota salvos para {file_name}.')

    def plot_spectrograms(self) -> None:
        """
        Método responsável por plotar os espectrogramas dos áudios carregados.
        """
        for idx, spectrogram in enumerate(self.spectrograms):
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(spectrogram, sr=self.sample_rates[idx], x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Espectrograma do arquivo {self.file_paths[idx]} (Nota: {self.notes[idx]})')
            plt.show()

    def get_data_for_training(self, regenerate: bool = False) -> Tuple[List[np.ndarray], List[str]]:
        """
        Gera os espectrogramas e retorna junto com os nomes das notas musicais. Se já existirem salvos, carrega-os.
        :param regenerate: Se True, regera os espectrogramas e notas, mesmo que já existam.
        :return: Tupla (lista de espectrogramas, lista de nomes de notas)
        """
        self.load_wavs(regenerate=regenerate)
        return self.spectrograms, self.notes
