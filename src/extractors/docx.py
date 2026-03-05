import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import docx
from docx.document import Document
from docx.table import Table, _Cell

from src.extractors.base import BaseExtractor, ExtractionState
from src.schemas.licitacao import ItemLicitacao

logger = logging.getLogger(__name__)


class DocxExtractor(BaseExtractor):
    """
    Classe responsável por extrair dados de arquivos .DOCX usando a biblioteca python-docx.
    """

    # =========================================================================
    # CONSTANTES ESPECÍFICAS DE DOCX
    # =========================================================================

    # Máximo de caracteres em uma célula para ser considerada parte de um título de coluna
    MAX_HEADER_CELL_LENGTH = 60

    # Agrupamento lógico das palavras-chave para busca de cabeçalho na linha inteira concatenada
    HEADER_KEYWORD_GROUPS = (
        ("item", "código"),
        ("objeto", "descri", "especific"),
        ("quant", "qtd"),
        ("unid", "und"),
    )

    # Tamanho mínimo exigido para resgatar uma descrição perdida usando a maior string da linha
    MIN_DESC_FALLBACK_LEN = 15

    # Tamanho máximo que uma string pode ter para ser avaliada como uma possível quantidade
    MAX_QTD_CELL_LEN = 20

    # Tamanho máximo para fazer uma busca agressiva de números na célula
    MAX_QTD_STRICT_LEN = 10

    # Unidades de fornecimento explícitas usadas no resgate visual caso a coluna da unidade suma
    VALID_UNIDS_SET = frozenset(
        [
            "UND",
            "UNID",
            "UNIDADE",
            "CX",
            "CAIXA",
            "KG",
            "L",
            "LITRO",
            "LITROS",
            "PCT",
            "PACOTE",
            "PAR",
            "PARES",
            "M",
            "METRO",
            "M2",
            "M3",
            "MÊS",
            "MESES",
            "SERVIÇO",
            "SERVICO",
            "CJ",
            "CONJUNTO",
            "KIT",
            "GALÃO",
            "FRASCO",
            "HORA SERVIÇO TECNICO",
            "ROLO",
            "TONELADA",
            "TON",
            "JG",
            "JOGO",
            "PEÇA",
            "PÇ",
            "UND SERVIÇO TÉCNICO",
        ]
    )

    # =========================================================================
    # EXTRAÇÃO E MÉTODOS AUXILIARES
    # =========================================================================

    def extract(self, file_path: Path) -> list[ItemLicitacao]:
        """
        Extrai os itens de licitação de um arquivo DOCX.

        Args:
            file_path (Path): Caminho do arquivo DOCX a ser processado.

        Returns:
            list[ItemLicitacao]: Lista de itens extraídos do DOCX.

        Raises:
            Exception: Qualquer erro durante a leitura ou processamento do arquivo.
        """
        if not file_path.exists() or not file_path.is_file():
            logger.error(f"Arquivo não encontrado ou inválido: {file_path}")
            return []

        extracted_items = []

        try:
            doc = docx.Document(str(file_path))
            logger.info(f"Iniciando DOCX: {file_path.name}")

            # Estado para regras de negócio entre linhas/páginas
            state: ExtractionState = {
                "last_lote": None,
                "item_counter": 1,
                "current_header_map": None,
                "pending_broken_desc": None,
                "last_extracted_item": None,
                "pending_item_num": None,
            }

            # Itera sobre todas as tabelas (incluindo aninhadas)
            for i, table in enumerate(self._iter_tables(doc)):
                # Converte objeto Table do Word para matriz de strings limpa
                table_data = self._table_to_list(table)

                if not table_data:
                    continue

                # Processa os dados da tabela
                items = self._process_table_data(table_data, state, table_index=i)
                extracted_items.extend(items)

            logger.info(
                f"Concluído DOCX: {file_path.name}. Itens brutos extraídos: {len(extracted_items)}"
            )

        except Exception as e:
            logger.error(f"Falha crítica ao ler o .DOCX '{file_path.name}': {e}", exc_info=True)

        return self._deduplicate_items(extracted_items)

    def _iter_tables(self, container: Document | _Cell) -> Iterator[Table]:
        """
        Generator recursivo para encontrar tabelas, inclusive aninhadas.

        Args:
            container (Document | _Cell): Pode ser um documento, célula ou tabela.

        Yields:
            Table: Tabelas encontradas no documento.
        """
        for table in container.tables:
            yield table
            # Procura tabelas aninhadas dentro das células
            for row in table.rows:
                for cell in row.cells:
                    yield from self._iter_tables(cell)

    def _table_to_list(self, table: Table) -> list[list[str]]:
        """
        Converte uma tabela do python-docx para uma lista de listas de strings.

        Args:
            table (Table): Objeto Table do python-docx.

        Returns:
            list[list[str]]: Matriz de strings representando o conteúdo da tabela.
        """
        data = []
        for row in table.rows:
            # List comprehension para extrair texto
            row_data = [cell.text.replace("\n", " ").strip() for cell in row.cells]

            if any(row_data):
                data.append(row_data)

        return data

    def _process_table_data(
        self, data: list[list[str]], state: ExtractionState, table_index: int
    ) -> list[ItemLicitacao]:
        """
        Identifica cabeçalhos e extrai itens de uma matriz de dados.

        Args:
            data (list[list[str]]): Matriz de strings representando o conteúdo da tabela.
            state (ExtractionState): Dicionário para manter estado entre linhas/páginas.
            table_index (int): Índice da tabela atual no documento.

        Returns:
            list[ItemLicitacao]: Lista de itens extraídos da tabela.
        """
        items = []
        current_lote = None

        for row in data:
            # Verifica se a linha é uma declaração de Lote. Se for, atualiza o estado e pula
            lote_found = self._extract_lote_from_row(row)

            if lote_found:
                current_lote = lote_found
                logger.debug(f"Lote detectado na tabela {table_index}: {current_lote}")
                state["last_lote"] = current_lote
                continue

            # Verifica se a linha é um Cabeçalho. Se for, mapeia as colunas e pula
            candidate_map = self._identify_columns(row)
            if candidate_map:
                current_map = state["current_header_map"]

                # Se for novo header (não repetido), atualiza o estado
                if not (current_map and self._is_repeated_header(row, current_map)):
                    state["current_header_map"] = candidate_map
                    logger.debug(f"Novo Header DOCX adotado: {candidate_map}")
                continue

            # Se já há um cabeçalho mapeado, tenta extrair a linha como um Item
            if state["current_header_map"]:
                item = self._create_item_from_row(
                    row, state["current_header_map"], state, current_lote
                )

                if item:
                    items.append(item)

        return items

    def _create_item_from_row(
        self,
        row: list[str],
        header_map: dict[str, int],
        state: ExtractionState,
        current_lote: str | None,
    ) -> ItemLicitacao | None:
        """
        Cria um ItemLicitacao a partir de uma linha de dados, usando o mapeamento do header e o
        estado para aplicar regras de negócio como resgate de lote e controle de item.

        Args:
            row (list[str]): Linha de dados representando um possível item.
            header_map (dict[str, int]): Mapeamento de colunas identificado no header.
            state (ExtractionState): Dicionário para manter estado entre linhas/páginas.
            current_lote (str | None): Lote detectado na linha atual.

        Returns:
            ItemLicitacao | None: Item extraído da linha ou None se não for possível extrair um
            item válido.
        """
        active_lote = current_lote if current_lote else state["last_lote"]
        temp_state = state.copy()
        temp_state["last_lote"] = active_lote

        item = self._parse_row_docx(row, header_map, temp_state)

        if not item:
            return None

        # Garante que cláusulas do edital e leis citadas na tabela não virem produtos fantasma
        desc_lower = item.objeto.lower()

        if desc_lower.startswith(self.INVALID_DESC_PREFIXES):
            return None

        if not item.lote and active_lote:
            item.lote = active_lote

        # Se o item foi criado com sucesso, efetiva as mudanças no estado global
        state["item_counter"] = temp_state["item_counter"]
        return item

    def _parse_row_docx(
        self, row: Sequence[str | None], mapping: dict[str, int], state: ExtractionState
    ) -> ItemLicitacao | None:
        """
        Processa uma linha de tabela de arquivo Word (DOCX).
        Implementa defesas contra deslocamento de colunas e células invisíveis mescladas.

        Args:
            row (Sequence[str | None]): Linha de texto representando os dados.
            mapping (dict): Mapeamento de colunas identificado no header.
            state (ExtractionState): Dicionário para manter estado entre linhas/páginas.

        Returns:
            ItemLicitacao | None: Item extraído da linha ou None se não for possível extrair um
            item válido.
        """
        try:
            # Extração da Descrição
            desc = self._extract_descricao(row, mapping)
            if not desc:
                return None

            # Extração da Quantidade
            qtd = self._extract_quantidade(row, state["item_counter"])
            if qtd is None:
                return None

            # Item e Lote (Controle de Estado)
            idx_item = mapping.get("item")
            final_item = self._update_item_counter(row, idx_item, state)
            final_lote = self._update_lote_state(row, mapping.get("lote"), state)

            # Extração da Unidade de Fornecimento
            unid = self._extract_unidade_fornecimento(row, mapping)

            return ItemLicitacao(
                item=final_item,
                quantidade=qtd,
                objeto=desc,
                unidade_fornecimento=unid,
                lote=final_lote,
            )

        except Exception as e:
            # Registra um pedaço da linha que causou a falha
            row_sample = str(row)[:60] + "..." if row else "Linha Vazia"

            logger.debug(
                f"Erro inesperado no parser DOCX. Linha ({row_sample}): {e}", exc_info=True
            )
            return None

    def _identify_columns(self, row: Sequence[str | None]) -> dict[str, int] | None:
        """
        Identifica o mapeamento de colunas em uma linha de cabeçalho.

        Sobrescreve o método base para adicionar uma proteção específica para DOCX:
        aborta a identificação se alguma célula contiver um texto muito longo,
        indicando tratar-se da descrição de um produto e não de um título de coluna.

        Args:
            row (Sequence[str | None]): Linha de texto representando um possível cabeçalho.

        Returns:
            dict[str, int] | None: Dicionário com o mapeamento das colunas ou None se
            não for um cabeçalho válido.
        """
        row_str_raw = [str(c).strip() for c in row if c]

        if any(len(c) > self.MAX_HEADER_CELL_LENGTH for c in row_str_raw):
            return None

        # Se passou na proteção, chama o método base para identificar as colunas normalmente
        return super()._identify_columns(row)

    def _is_repeated_header(self, row: Sequence[str | None], header_map: dict[str, int]) -> bool:
        """
        Verifica se a linha atual é um cabeçalho repetido da tabela.

        Sobrescreve o método da classe base porque, no formato DOCX, as colunas
        podem sofrer deslocamento ou mesclagem invisível.

        Args:
            row (Sequence[str | None]): Sequência de strings representando as células da linha.
            header_map (dict[str, int]): Mapeamento de colunas (mantido na assinatura pelo
            Polimorfismo).

        Returns:
            bool: True se a linha contiver grupos de palavras suficientes para ser considerada
            um cabeçalho, False caso contrário.
        """
        matches = 0
        row_str = " ".join(str(c).lower() for c in row if c)

        for keyword_group in self.HEADER_KEYWORD_GROUPS:
            if any(keyword in row_str for keyword in keyword_group):
                matches += 1

        return matches >= self.HEADER_MATCH_THRESHOLD

    def _extract_descricao(self, row: Sequence[str | None], mapping: dict[str, int]) -> str | None:
        """
        Extrai a descrição, com heurística de fallback para a maior string da linha caso a coluna
        deslize.

        Args:
            row (Sequence[str | None]): Linha de texto representando os dados.
            mapping (dict): Mapeamento de colunas identificado no header.

        Returns:
            str | None: Descrição extraída ou resgatada, ou None se não for possível extrair uma
            descrição válida.
        """
        idx_desc = mapping.get("objeto")
        desc = self._get_text_safe(row, idx_desc)

        # Se não achou na coluna certa, pega o texto mais longo da linha
        if not desc or len(desc) < self.MIN_DESC_FALLBACK_LEN:
            maior_texto = ""
            for cell in row:
                if cell and len(str(cell)) > len(maior_texto):
                    maior_texto = str(cell)

            if len(maior_texto) > self.MIN_DESC_FALLBACK_LEN:
                desc = maior_texto

        if not desc or len(desc) < self.MIN_DESC_LEN:
            return None

        return desc

    def _extract_quantidade(self, row: Sequence[str | None], expected_item: int) -> int | None:
        """
        Busca a quantidade na linha, buscando ignorar falsos positivos.

        Args:
            row (Sequence[str | None]): Linha de texto representando os dados.
            expected_item (int): O número do item esperado, para aplicar a heurística
                anti-deslizamento de colunas.

        Returns:
            int | None: Quantidade extraída ou None se não for possível extrair.
        """
        found_nums = []

        # Varre a linha inteira buscando números, mas ignora células que parecem conter datas,
        # valores ou anexos
        for cell in row:
            cstr = str(cell).strip() if cell else ""

            if (
                len(cstr) > self.MAX_QTD_CELL_LEN
                or re.search(r"\d{2}/\d{2}/\d{4}", cstr)
                or "R$" in cstr.upper()
                or "ANEXO" in cstr.upper()
            ):
                continue

            # Primeiro tenta buscar uma célula que deve conter apenas um número (com ou sem pontos)
            nums = re.findall(r"^\s*(\d+)\s*$", cstr.replace(".", ""))

            # Se não encontrar, tenta buscar qualquer número na célula, se a célula for
            # relativamente curta
            if not nums and len(cstr) <= self.MAX_QTD_STRICT_LEN:
                nums = re.findall(r"\d+", cstr.replace(".", ""))

            if nums:
                num_val = int(nums[0])
                if num_val > 0:
                    found_nums.append(num_val)

        if not found_nums:
            return None

        # Se encontrou mais de um número, remove o número do item esperado
        if len(found_nums) > 1 and expected_item in found_nums:
            found_nums.remove(expected_item)

        qtd = found_nums[0]
        return qtd if qtd > 0 else None

    def _extract_unidade_fornecimento(
        self, row: Sequence[str | None], mapping: dict[str, int]
    ) -> str:
        """
        Extrai a unidade de fornecimento, com resgate caso a coluna original tenha desaparecido.

        Primeiro tenta extrair da coluna mapeada. Se o resultado for o padrão genérico
        "Unidade", faz uma segunda passada procurando por unidades válidas no Set de referência.

        Args:
            row (Sequence[str | None]): Linha de texto representando os dados.
            mapping (dict[str, int]): Mapeamento de colunas identificado no cabeçalho.

        Returns:
            str: Unidade de fornecimento extraída ou resgatada. Padrão: "Unidade".
        """
        idx_unid = mapping.get("unidade_fornecimento")
        unid_raw = self._get_text_safe(row, idx_unid)
        unid = self._clean_unidade_fornecimento(unid_raw)

        # Se ficou com o padrão genérico, varre a linha procurando uma unidade válida no Set
        if unid.upper() == "UNIDADE":
            for cell in row:
                if cell:
                    cstr = str(cell).strip().upper()

                    if cstr in self.VALID_UNIDS_SET:
                        return cstr.capitalize()

        return unid
