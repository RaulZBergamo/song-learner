"""
Módulo para cuidar das interações com o Hugging Face.
"""

import os
import logging

from datasets import Dataset, Audio, load_dataset
from huggingface_hub import list_datasets

class HugginfaceRepository():
    """
    Classe para cuidar das interações com o Hugging Face.
    """

    def __init__(self, hugface_user: str) -> None:
        """
        Instancia um novo objeto HugginfaceRepository.

        :param hugface_user: Usuário do Hugging Face.
        """
        self.hugface_user = hugface_user

    def __check_repo_name(self, repo_name: str) -> str:
        """
        Verifica se o nome do repositório é válido.
        """
        if not repo_name.startswith(self.hugface_user):
            logging.warning(
                "O nome do repositório não começa com o nome do usuário do Hugging Face. "
                "Adicionando o nome do usuário ao nome do repositório."
            )
            repo_name = f"{self.hugface_user}/{repo_name}"

        return repo_name

    def upload_dataset_to_huggingface(
        self,
        dataset_path: str,
        repo_name: str,
        private: bool = True
    ) -> Dataset:
        """
        Faz upload do dataset para o Hugging Face Dataset Hub.

        :param dataset_path: Caminho do dataset (estrutura contendo os arquivos).
        :param repo_name: Nome do repositório no Hugging Face Hub.
        :param private: Se True, o repositório será privado.
        """
        repo_name = self.__check_repo_name(repo_name)

        audio_files = []
        labels = []
        
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    audio_files.append(os.path.join(root, file))
                    labels.append(os.path.basename(root))

        data_dict = {
            "audio": audio_files,
            "label": labels
        }
        dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
        
        dataset.push_to_hub(repo_name, private=private)

        logging.info("Dataset '%s' enviado com sucesso!", repo_name)

        return dataset

    def get_dataset_from_huggingface(self, repo_name: str) -> Dataset:
        """
        Baixa o dataset do Hugging Face Dataset Hub.

        :param repo_name: Nome do repositório no Hugging Face Hub.
        :return: O dataset baixado.
        """
        repo_name = self.__check_repo_name(repo_name)

        logging.info("Baixando dataset '%s'...", repo_name)

        dataset = load_dataset(path=repo_name)

        logging.info("Dataset '%s' baixado com sucesso!", repo_name)
        logging.info("Número de exemplos: %s", dataset)

        return dataset

    def check_existing_datasets(self, repo_name: str) -> bool:
        """
        Lista os datasets existentes no Hugging Face Dataset Hub.

        :param repo_name: Nome do repositório no Hugging Face Hub.
        :return: True se o dataset já existe, False caso contrário.
        """
        repo_name = self.__check_repo_name(repo_name)

        datasets = list_datasets(author=self.hugface_user)
        datasets = [dataset for dataset in datasets]

        if any(
            dataset.id == repo_name for dataset in datasets
        ):
            logging.info("Dataset '%s' já existe.", repo_name)
            return True
        
        logging.info("Dataset '%s' não existe.", repo_name)
        return False
