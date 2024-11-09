"""
Módulo para obter os dados de áudio a serem utilizados.
"""

import logging
import os
import pickle
from typing import Tuple, List

import numpy as np
import librosa
import librosa.display
from midi.midi_converter import MidiConverter

class WavController:
    """
    Classe responsável por obter os dados de áudio e 
    gerar espectrogramas e notas musicais para treinamento.
    """

    def __init__(
        self,
        file_paths: List[str],
        midi_converter: MidiConverter,
        save_path: str = None
    ) -> None:
        """
        Instancia um novo objeto WavController.
        :param file_paths: Lista de caminhos de arquivos de áudio.
        :param midi_converter: Instância de MidiConverter para converter pitches em notas musicais.
        :param save_path: Caminho para salvar os espectrogramas e notas (opcional).
        """
        self.file_paths = file_paths
        self.midi_converter = midi_converter
        self.save_path = save_path or os.getcwd() + '/spectrograms/'
        os.makedirs(self.save_path, exist_ok=True)  # Cria o diretório se não existir

        self.spectrograms = []
        self.notes = []

    def load_wavs(self, regenerate: bool = False) -> None:
        """
        Método responsável por carregar arquivos de áudio .wav 
        Ou carregar espectrogramas e notas salvas.
        :param regenerate: Se True, regera os dados, mesmo que existam salvos.
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
                logging.info('Espectrograma e nota carregados para %s.', file_name)
            else:
                # Carregar o arquivo de áudio e gerar os dados
                logging.info('Carregando e gerando espectrograma para %s.', file_name)
                self._process_file(file_path)
                self._save_data(file_name)

    def _process_file(self, file_path: str) -> None:
        """
        Processa um único arquivo de áudio: gera espectrograma e converte o pitch para nome de nota.
        :param file_path: Caminho do arquivo .wav.
        """
        audio_data, _ = librosa.load(file_path, sr=16000)

        # Gera o espectrograma
        spectogram_abs = np.abs(librosa.stft(audio_data))
        spectrogram = librosa.amplitude_to_db(spectogram_abs, ref=np.max)
        self.spectrograms.append(spectrogram)

        # Extrai o pitch do nome do arquivo e converte para o nome da nota
        pitch = self._extract_pitch_from_filename(file_path)
        note_name = self.midi_converter.midi_to_note_name(pitch)
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
            logging.info('Pitch extraído do nome do arquivo %s: %d.', file_path, pitch)
            return pitch
        except Exception as e:
            logging.error('Erro ao extrair pitch do nome do arquivo %s. Error: %s', file_path, e)
            raise ValueError(
                f'Nome do arquivo {file_path} não está no formato esperado.'
            ) from e

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

        logging.info('Espectrograma e nota salvos para %s.', file_name)

    def get_data(self, regenerate: bool = False) -> Tuple[List[np.ndarray], List[str]]:
        """
        Gera os espectrogramas e retorna junto com os nomes das notas musicais.
        :param regenerate: Se True, regera os espectrogramas e notas, mesmo que já existam.
        :return: Tupla (lista de espectrogramas, lista de nomes de notas)
        """
        self.load_wavs(regenerate=regenerate)
        return self.spectrograms, self.notes
