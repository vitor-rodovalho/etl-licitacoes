import json
import logging
import tempfile
import uuid
import zipfile
from pathlib import Path

import pdfplumber
from unidecode import unidecode

from src.extractors.base import BaseExtractor
from src.extractors.docx import DocxExtractor
from src.extractors.pdf import PDFExtractor
from src.schemas.licitacao import ItemLicitacao, Licitacao

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Classe responsável por orquestrar o processo de leitura dos arquivos JSON, encontrar as pastas
    de anexos, e delegar a extração dos dados para os extratores adequados, construindo o objeto
    Licitacao final.
    """

    # =========================================================================
    # CONSTANTES ESPECÍFICAS DO ORQUESTRADOR
    # =========================================================================

    # Quantidade de pontos de prioridade que finaliza a busca por outros arquivos
    FILE_PRIORITY_BREAK = 100

    # Relação dos nomes dos arquivos e seus respectivos pontos atribuídos
    NAME_PRIORITY_RULES = (
        ("relacaoitens", 100),
        ("termo", 75),
        ("edital", 50),
    )

    # Relação do conteúdos dos arquivos e seus respectivos pontos atribuídos
    CONTENT_PRIORITY_RULES = (
        ("relacao de itens", 100),
        ("termo de referencia", 75),
        ("edital de licitacao", 50),
        ("edital de pregao", 50),
    )

    # =========================================================================
    # MÉTODOS, PROCESSAMENTO E REGRAS DE NEGÓCIO
    # =========================================================================

    def __init__(self):
        """
        Inicializa os extratores suportados.
        """

        # Inicializa os extratores uma única vez
        self.extractors = {".pdf": PDFExtractor(), ".docx": DocxExtractor()}

    def process_directory(self, input_dir: Path) -> list[Licitacao]:
        """
        Varre o diretório buscando pares de JSON + Pasta de Anexos.

        Args:
            input_dir (Path): Diretório raiz onde estão os arquivos JSON e as pastas de anexos.

        Returns:
            list[Licitacao]: Lista de objetos Licitacao processados.
        """

        results = []
        # Encontra todos os JSONs na pasta raiz
        json_files = list(input_dir.glob("*.json"))

        logger.info(f"Encontrados {len(json_files)} arquivos de licitação para processar.")

        for json_file in json_files:
            try:
                licitacao = self._process_single_licitacao(json_file)
                results.append(licitacao)
            except Exception as e:
                logger.error(f"Falha crítica ao processar {json_file.name}: {e}")

        return results

    def _process_single_licitacao(self, json_path: Path) -> Licitacao:
        """
        Lê o JSON, encontra a pasta de anexos correspondente e processa os arquivos compatíveis.

        Args:
            json_path (Path): Caminho para o arquivo JSON da licitação.

        Returns:
            Licitacao: Objeto contendo os dados extraídos da licitação.
        """

        logger.info(f"=== Processando JSON: {json_path.name} ===")

        # Lê Metadados do JSON
        with json_path.open(mode="r", encoding="utf-8") as f:
            data = json.load(f)

        # Extrai campos básicos para o esquema final
        meta = data.get("data", {})

        # Encontra pasta de anexos
        attachments_dir = json_path.with_suffix("")

        processed_files = []
        all_items = []

        if attachments_dir.exists() and attachments_dir.is_dir():
            # Verifica se existe algum ZIP no diretório
            has_zips = any(
                f.suffix.lower() == ".zip" for f in attachments_dir.iterdir() if f.is_file()
            )

            # Se não houver ZIPs, processa os arquivos diretamente
            if not has_zips:
                valid_files = [
                    f
                    for f in attachments_dir.iterdir()
                    if f.is_file()
                    and not f.name.startswith("._")
                    and f.suffix.lower() in self.extractors
                    and f.stat().st_size > 0
                ]
                self._run_extraction(valid_files, all_items, processed_files)

            # Se houver ZIPs, cria um ambiente temporário para descompactar e processar os arquivos
            else:
                logger.info("Arquivos ZIP detectados. Iniciando ambiente temporário.")

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir_path = Path(temp_dir)

                    # Obtém os arquivos originais e os arquivos descompactados válidos
                    valid_files = self._unzip_files(attachments_dir, temp_dir_path)

                    # Roda o processamento enquanto o temp_dir ainda está vivo
                    self._run_extraction(valid_files, all_items, processed_files)
        else:
            logger.warning(f"Pasta de anexos não encontrada: {attachments_dir}")

        # Remove itens duplicados que vieram de múltiplos anexos
        if all_items and self.extractors:
            first_extractor = next(iter(self.extractors.values()))
            all_items = first_extractor._deduplicate_items(all_items)

        if not all_items:
            logger.warning(f"Finalizado {json_path.name}: 0 itens extraídos.")
        else:
            logger.info(f"Finalizado {json_path.name}: {len(all_items)} itens consolidados.")

        logger.info("-" * 50)

        # Constrói e retorna o objeto Licitacao final
        return Licitacao(
            arquivo_json=json_path.name,
            numero_pregao=str(meta.get("numero_pregao", "")),
            orgao=str(meta.get("orgao", "")),
            cidade=str(meta.get("cidade", "")),
            estado=str(meta.get("estado", "")),
            anexos_processados=processed_files,
            itens_extraidos=all_items,
        )

    def _run_extraction(
        self, files: list[Path], all_items: list[ItemLicitacao], processed_files: list[str]
    ) -> None:
        """
        Recebe uma lista de arquivos válidos, calcula a prioridade e extrai os itens.

        Args:
            files (list[Path]): Lista de arquivos a serem processados.
            all_items (list[ItemLicitacao]): Lista onde os itens extraídos serão acumulados.
            processed_files (list[str]): Lista onde os nomes dos arquivos processados serão
            acumulados.
        """

        if not files:
            return

        # Calcula a prioridade de cada arquivo e armazena em uma lista de tuplas (score, file_path)
        files_with_priority = []
        for f in files:
            score = self._calculate_file_priority(f)
            files_with_priority.append((score, f))

        # Ordena pelo primeiro elemento da tupla (o Score)
        files_with_priority.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"Ordem de prioridade: {[f.name for _, f in files_with_priority]}")

        for priority, file_path in files_with_priority:
            extractor = self._get_extractor(file_path)

            if extractor:
                # Processa o arquivo e extrai os itens
                try:
                    items = extractor.extract(file_path)
                    if items:
                        all_items.extend(items)
                        processed_files.append(file_path.name)

                        if priority >= self.FILE_PRIORITY_BREAK:
                            logger.info(
                                f"Documento de alta relevância encontrado "
                                f"({file_path.name}). Finalizando busca."
                            )
                            break
                except Exception as e:
                    logger.warning(f"Falha ao processar arquivo {file_path.name}: {e}")

    def _unzip_files(self, source_dir: Path, temp_work_dir: Path) -> list[Path]:
        """
        Descompacta arquivos ZIP encontrados no diretório de anexos para um diretório temporário.

        Args:
            source_dir (Path): Diretório onde estão os arquivos ZIP.
            temp_work_dir (Path): Diretório temporário onde os arquivos serão descompactados.

        Returns:
            list[Path]: Lista de arquivos válidos encontrados após a descompactação.
        """

        valid_files = []

        def _scan_directory(current_dir: Path, current_depth: int = 0, max_depth: int = 3):
            """
            Varre o diretório atual procurando por arquivos válidos e ZIPs. Se encontrar um ZIP,
            descompacta e escaneia recursivamente, respeitando o limite de profundidade para evitar
            loops infinitos.

            Args:
                current_dir (Path): Diretório a ser escaneado.
                current_depth (int): Profundidade atual da recursão.
                max_depth (int): Limite máximo de profundidade para escaneamento.
            """

            if current_depth > max_depth:
                logger.warning(
                    f"Limite de profundidade ({max_depth}) atingido. Ignorando {current_dir.name}"
                )
                return

            for f in current_dir.iterdir():
                # Ignora arquivos ocultos, arquivos corrompidos e não arquivos
                if f.name.startswith("._") or not f.is_file() or f.stat().st_size == 0:
                    continue

                extension = f.suffix.lower()

                # Se for um arquivo válido conforme o tipo, adiciona à lista
                if extension in self.extractors:
                    valid_files.append(f)

                # Se for um zip, descompacta e escaneia recursivamente
                elif extension == ".zip":
                    logger.info(f"Descompactando ZIP: {f.name} (Nível {current_depth})")
                    extract_path = temp_work_dir / f"{f.stem}_{uuid.uuid4().hex[:8]}"
                    extract_path.mkdir(exist_ok=True)

                    try:
                        with zipfile.ZipFile(f, "r") as zip_ref:
                            zip_ref.extractall(extract_path)

                        # Recursão
                        _scan_directory(extract_path, current_depth + 1, max_depth)

                    except zipfile.BadZipFile:
                        logger.error(f"Arquivo ZIP corrompido ou inválido: {f.name}")
                    except Exception as e:
                        logger.error(f"Erro ao extrair ZIP {f.name}: {e}")

        _scan_directory(source_dir)

        return valid_files

    def _get_extractor(self, file_path: Path) -> BaseExtractor | None:
        """
        Retorna o extrator adequado baseado na extensão, ou None caso a extensão não seja suportada.

        Args:
            file_path (Path): Caminho do arquivo a ser processado.

        Returns:
            BaseExtractor | None: Instância do extrator ou None se não suportado.
        """
        return self.extractors.get(file_path.suffix.lower())

    def _calculate_file_priority(self, file_path: Path) -> int:
        """
        Calcula um score de prioridade baseado no nome do arquivo e conteúdo.
        Quanto maior o número, antes o arquivo deve ser processado.

        Args:
            file_path (Path): Caminho do arquivo a ser avaliado.

        Returns:
            int: Score de prioridade (0-100).
        """
        # Verifica prioridade pelo nome do arquivo
        name = unidecode(file_path.name.lower())

        for keyword, score in self.NAME_PRIORITY_RULES:
            if keyword in name:
                return score

        # Se não for PDF, não é possível ler o conteúdo, então encerra com prioridade mínima
        if file_path.suffix.lower() != ".pdf":
            return 0

        # Verifica prioridade espiando o conteúdo da página 1
        try:
            with pdfplumber.open(file_path) as pdf:
                first_page = pdf.pages[0].extract_text() if pdf.pages else None

            # Se conseguiu extrair algum texto, aplica as regras de conteúdo
            if first_page:
                text_clean = unidecode(first_page.lower())

                for keyword, score in self.CONTENT_PRIORITY_RULES:
                    if keyword in text_clean:
                        logger.debug(f"[{file_path.name}] '{keyword}' identificada na página 1.")
                        return score

        except Exception as e:
            logger.debug(f"Não foi possível espiar o conteúdo de {file_path.name}: {e}")

        # Fallback de segurança se nenhuma das buscas retornar alguma prioridade
        return 0
