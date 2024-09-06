"""
Módulo para obter os dados de audio a serem utilizados
Por enquanto vamos fazer o uso de um dataset de músicas MIDI
E vamos fazer a obtenção dos dados por um link de download
"""

import logging
import tarfile
import os
from tqdm import tqdm

import requests

class DataSet:
    """
    Classe responsável por obter os dados de audio a serem utilizados no projeto.
    """

    def __init__(self, data_set_url: str, download_path: str = None) -> None:
        """
        Instancia um novo objeto DataSet.
        :param data_set_url: URL do dataset a ser utilizado.
        :param download_url: URL para download do dataset.
        """

        if not data_set_url:
            raise ValueError('O link do dataset não pode ser vazio.')

        self.download_url = data_set_url
        self.download_path = download_path or os.getcwd() + "/assets/"

        self.file_path = f"{self.download_path}/{self.download_url.split('/')[-1]}"
        self.extracted_path = f"{self.download_path}/dataset/"
        self.audios_path = f"{self.extracted_path}/audios/"

    def get_data_set(self) -> None:
        """
        Método responsável por obter o dataset.
        """
        logging.info('Obtendo dataset...')

        if os.path.exists(self.file_path):
            logging.info('Dataset já existe.')
            self.__uncompress_data_set()

        else:
            self.__download_data_set()

        self.__validate_data_set()

    def __download_data_set(self) -> None:
        """
        Método responsável por fazer o download do dataset com barra de progresso.
        """
        logging.info('Baixando dataset...')

        os.makedirs(self.download_path, exist_ok=True)

        try:
            with requests.get(self.download_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))  # Obtém o tamanho do arquivo
                block_size = 8192  # Tamanho do chunk (8KB)

                with open(self.file_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=self.file_path
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=block_size):
                        f.write(chunk)
                        pbar.update(len(chunk))

            logging.info('Dataset baixado com sucesso no caminho: %s', self.file_path)
            self.__uncompress_data_set()

        except requests.HTTPError as e:
            logging.error(
                'Erro ao baixar dataset. Status code: %s. Message %s',
                e.response.status_code,
                e.response.text
            )
            raise

    def __uncompress_data_set(self) -> None:
        """
        Método responsável por descompactar o dataset.
        Ele é baixado com a extensão .tar.gz
        """
        if os.path.exists(self.extracted_path):
            logging.info('Dataset já descompactado.')
            return

        logging.info('Descompactando dataset...')

        if not os.path.exists(self.file_path):
            logging.error('Dataset não encontrado.')
            raise FileNotFoundError('Dataset não encontrado.')

        with tarfile.open(self.file_path) as tar:
            tar.extractall(self.extracted_path)

        logging.info('Dataset descompactado com sucesso no caminho: %s', self.download_path)

    def __validate_data_set(self) -> None:
        """
        Método responsável por validar o dataset.
        Nesse caso, vamos verificar se o dataset foi descompactado
        E se existem arquivos de audio.
        """
        if not os.path.exists(self.extracted_path):
            logging.error('Dataset não encontrado.')
            raise FileNotFoundError('Dataset não encontrado.')

        if not os.listdir(self.extracted_path):
            logging.error('Dataset vazio.')
            raise FileNotFoundError('Dataset vazio.')

        if not os.listdir(self.audios_path):
            logging.error('Nenhum audio encontrado.')
            raise FileNotFoundError('Nenhum audio encontrado.')

    def filter_data_set(self, filter_by: str) -> None:
        """
        Método responsável por filtrar o dataset.
        :param filter_by: Parâmetro para filtrar o dataset.
        """
