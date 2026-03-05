import json
import logging
from pathlib import Path

from src.services.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Serviço responsável por encapsular o fluxo completo de ETL (Extract, Transform, Load).
    Conecta a entrada (CLI), o processamento (Orchestrator) e a saída (JSON).
    """

    def run(self, input_path: Path, output_path: Path) -> None:
        """
        Função principal que orquestra a execução do pipeline de processamento de licitações.

        Args:
            input_path (Path): Caminho para a pasta contendo os arquivos JSON e pastas com anexos.
            output_path (Path): Caminho completo onde o arquivo array JSON final será salvo.
        """

        logger.info("=== Iniciando Pipeline de Extração ===")

        # Validação de regras de negócio
        self._validate_input(input_path)

        # Execução da lógica principal
        orchestrator = Orchestrator()
        logger.info(f"Delegando processamento para o Orchestrator em: {input_path}")

        resultados = orchestrator.process_directory(input_path)

        logger.info(f"Orchestrator finalizou. {len(resultados)} licitações retornadas.")

        # Persistência dos dados
        self._save_results(resultados, output_path)

        logger.info("=== Pipeline finalizado com sucesso ===")

    def _validate_input(self, input_path: Path) -> None:
        """
        Valida se o input é processável.

        Args:
            input_path (Path): Caminho para a pasta contendo os arquivos JSON e pastas com anexos.

        Raises:
            FileNotFoundError: Se o diretório de entrada não existir.
            NotADirectoryError: Se o caminho informado não for uma pasta.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_path}")
        if not input_path.is_dir():
            raise NotADirectoryError(f"O caminho informado não é uma pasta: {input_path}")

        # Se a pasta estiver vazia
        if not any(input_path.iterdir()):
            logger.warning(f"Atenção: O diretório {input_path} está vazio.")

    def _save_results(self, data: list, output_path: Path) -> None:
        """
        Gerencia a escrita do arquivo final.

        Args:
            data (list): Lista de objetos a serem salvos.
            output_path (Path): Caminho completo onde o arquivo array JSON final será salvo.
        """
        # Garante que a pasta de destino exista
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Salvando arquivo final em: {output_path}")

        with output_path.open("w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in data], f, indent=2, ensure_ascii=False)
