"""
Módulo para obter os dados de áudio a serem utilizados.
"""

import logging
import os
from typing import Tuple, Dict, Any

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
        midi_converter: MidiConverter
    ) -> None:
        """
        Instancia um novo objeto WavController.
        :param file_paths: Lista de caminhos de arquivos de áudio.
        :param midi_converter: Instância de MidiConverter para converter pitches em notas musicais.
        :param save_path: Caminho para salvar os espectrogramas e notas (opcional).
        """
        self.midi_converter = midi_converter

    def load_wav(self, audio_data: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """
        Método responsável por carregar arquivos de áudio .wav 
        Ou carregar espectrogramas e notas salvas.
        :param regenerate: Se True, regera os dados, mesmo que existam salvos.
        """
        pitch = self.extract_pitch_from_filename(audio_data['path'])

        stft = librosa.stft(audio_data['array'], n_fft=2048, hop_length=512)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        return spectrogram, pitch

    def extract_pitch_from_filename(self, file_path: str) -> float:
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
            return float(pitch)
        except Exception as e:
            logging.error('Erro ao extrair pitch do nome do arquivo %s. Error: %s', file_path, e)
            raise ValueError(
                f'Nome do arquivo {file_path} não está no formato esperado.'
            ) from e
