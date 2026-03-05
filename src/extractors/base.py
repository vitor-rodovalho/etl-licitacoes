import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from unidecode import unidecode

from src.core.config import HEADER_MAPPING
from src.schemas.licitacao import ItemLicitacao

logger = logging.getLogger(__name__)


class ExtractionState(TypedDict):
    last_lote: str | None
    item_counter: int
    current_header_map: dict[str, int] | None
    pending_broken_desc: str | None
    last_extracted_item: ItemLicitacao | None
    pending_item_num: int | None


class BaseExtractor(ABC):
    """
    Classe base que centraliza a lógica de interpretação de tabelas e dados.
    """

    # =========================================================================
    # REGRAS DE ESTRUTURA E CABEÇALHO
    # =========================================================================

    # Quantidade mínima de colunas identificadas para confirmar que a linha é um cabeçalho
    HEADER_MATCH_THRESHOLD = 2

    # Mínimo de colunas validadas necessárias para inicializar o mapeamento do extrator
    MIN_COLUMNS_FOR_HEADER = 2

    # Tamanho mínimo da palavra na célula para aceitar um "match parcial" com o nome da coluna
    MIN_MATCH_LEN = 2

    # Colunas sensíveis que exigem validação rigorosa para não sofrerem falsos positivos
    RESTRICTED_COLUMN_KEYS = frozenset({"item", "quantidade"})

    # Termos (geralmente financeiros ou de unidade) que anulam a detecção das colunas restritas
    FORBIDDEN_COLUMN_TERMS = ("vlr", "valor", "preco", "preço", "total", "r$", "unit")

    # =========================================================================
    # REGRAS DE LIMPEZA E SANITIZAÇÃO GERAL
    # =========================================================================

    # Máximo de letras permitidas dentro de uma string para ser tentada como número no fallback
    MAX_ALPHA_IN_NUMBER = 2

    # Caracteres isolados que indicam descrições ou unidades corrompidas em tabelas mal formatadas
    GARBAGE_CHARS = frozenset({"-", "_", ".", " ", "x"})

    # Prefixos jurídicos/administrativos indicando que a linha é cláusula do edital, não um produto
    INVALID_DESC_PREFIXES = ("art.", "lei ", "decreto", "data:", "assinatura", "pregão", "processo")

    # =========================================================================
    # REGRAS DE NEGÓCIO DO DOMÍNIO
    # =========================================================================

    # Tamanho mínimo absoluto para uma string ser considerada uma descrição de produto válida
    MIN_DESC_LEN = 3

    # Tamanho máximo aceitável para o texto de uma Unidade de Fornecimento
    MAX_UNID_LEN = 20

    # Limite máximo de quantidade para evitar a extração acidental de códigos
    MAX_VALID_QUANTITY = 100000

    # Palavras-chave que indicam endereços, bloqueando a falsa detecção de um "Lote"
    LOTE_BLOCK_WORDS = frozenset(
        [
            "RUA",
            "QUADRA",
            "QDA",
            "BAIRRO",
            "CEP",
            "ENDEREÇO",
            "ENDERECO",
            "AVENIDA",
            "ALAMEDA",
            "RODOVIA",
        ]
    )

    # Unidades preteridas na deduplicação se houver uma alternativa mais específica no mesmo item
    GENERIC_UNITS = frozenset({"UNIDADE"})

    # =========================================================================
    # MÉTODOS PRINCIPAIS E AUXILIARES
    # =========================================================================

    @abstractmethod
    def extract(self, file_path: Path) -> list[ItemLicitacao]:
        pass

    def _clean_rows(self, rows: Sequence[Sequence[str | None]]) -> list[list[str]]:
        """
        Limpa uma matriz bruta de linhas, removendo quebras de linha indesejadas e descartando
        linhas que estejam completamente vazias.

        Args:
            rows (Sequence[Sequence[str | None]]): Matriz bidimensional extraída da tabela.

        Returns:
            list[list[str]]: Matriz bidimensional higienizada, contendo apenas strings.
        """
        cleaned = []

        for row in rows:
            # Substitui quebras de linha por espaço e remove espaços extras
            clean_row = [(cell.replace("\n", " ").strip() if cell else "") for cell in row]

            # Só adiciona a linha se houver pelo menos uma célula preenchida
            if any(clean_row):
                cleaned.append(clean_row)

        return cleaned

    def _get_text_safe(self, row: Sequence[str | None], idx: int | None) -> str:
        """
        Extrai o texto de uma célula da linha de forma segura, tratando índices inválidos
        e quebras de linha.

        Args:
            row (Sequence[str | None]): Linha de dados atual.
            idx (int | None): Índice da coluna a ser extraída.

        Returns:
            str: Texto limpo da célula ou string vazia se inválido.
        """
        if idx is not None and 0 <= idx < len(row):
            val = row[idx]

            if val is not None:
                return str(val).replace("\n", " ").strip()

        return ""

    def _identify_columns(self, row: Sequence[str | None]) -> dict[str, int] | None:
        """
        Varre uma linha da tabela buscando palavras-chave que definam os títulos das colunas
        (Item, Objeto, Quantidade, Unidade).

        Args:
            row (Sequence[str | None]): Linha de texto higienizada.

        Returns:
            dict[str, int] | None: Mapeamento {nome_da_coluna: indice}, ou None se inválido.
        """
        mapping = {}
        row_str = [unidecode(str(c).lower().strip()) if c else "" for c in row]

        # Busca o mapping do arquivo de configuração
        header_map = HEADER_MAPPING

        for key, synonyms in header_map.items():
            for i, cell in enumerate(row_str):
                if self._should_skip_column_match(key, cell):
                    continue

                if self._match_synonym(synonyms, cell):
                    mapping[key] = i
                    break

            # Se já achou a coluna para esta chave, pula para a próxima chave
            if key in mapping:
                continue

        # Para ser considerado um cabeçalho viável, a tabela precisa ter obrigatoriamente a coluna
        # "objeto" e pelo menos mais uma coluna auxiliar
        if "objeto" in mapping and len(mapping) >= self.MIN_COLUMNS_FOR_HEADER:
            return mapping

        return None

    def _should_skip_column_match(self, key: str, cell: str) -> bool:
        """
        Evita o falso positivo de confundir colunas financeiras com colunas de quantidade/item.

        Args:
            key (str): O nome da coluna que está sendo avaliada (ex: "item", "quantidade").
            cell (str): O conteúdo da célula que está sendo verificada.

        Returns:
            bool: True se a correspondência deve ser ignorada, False caso contrário.
        """
        return key in self.RESTRICTED_COLUMN_KEYS and any(
            v in cell for v in self.FORBIDDEN_COLUMN_TERMS
        )

    def _match_synonym(self, synonyms: Sequence[str], cell: str) -> bool:
        """
        Verifica se algum dos sinônimos está presente na célula de forma isolada.

        Args:
            synonyms (Sequence[str]): Lista de sinônimos a serem verificados.
            cell (str): O conteúdo da célula a ser verificada.

        Returns:
            bool: True se encontrar um sinônimo isolado, False caso contrário.
        """
        for syn in synonyms:
            if syn == cell:
                return True

            if syn in cell:
                idx = cell.find(syn)
                prev_char = cell[idx - 1] if idx > 0 else " "
                next_char = cell[idx + len(syn)] if (idx + len(syn)) < len(cell) else " "

                # Só considera match se o sinônimo não estiver engolido no meio de outra palavra
                if not prev_char.isalpha() and not next_char.isalpha():
                    return True
        return False

    def _is_repeated_header(self, row: Sequence[str | None], header_map: dict[str, int]) -> bool:
        """
        Verifica se a linha atual é uma repetição do cabeçalho da tabela, baseando-se na
        correspondência estrita dos índices de coluna mapeados.

        Args:
            row (Sequence[str | None]): Linha de dados atual.
            header_map (dict[str, int]): Mapeamento atual de colunas.

        Returns:
            bool: True se encontrar correspondências suficientes para ser considerado cabeçalho.
        """
        test_cols = [k for k in header_map if k in ["item", "objeto", "quantidade"]]
        matches = 0

        for col_name in test_cols:
            idx = header_map[col_name]

            if idx < len(row):
                cell_raw = row[idx]
                if cell_raw is not None:
                    cell_val = unidecode(str(cell_raw).lower())

                    # Correspondência exata ou match parcial se a string for maior que o mínimo
                    if col_name in cell_val or (
                        len(cell_val) > self.MIN_MATCH_LEN and cell_val in col_name
                    ):
                        matches += 1

        return matches >= self.HEADER_MATCH_THRESHOLD

    def _extract_lote_from_row(self, row: Sequence[str | None]) -> str | None:
        """
        Verifica se a linha contém uma declaração de Lote ou Grupo.
        Possui blindagem contra falsos positivos comuns, como endereços contendo a palavra "Lote".

        Args:
            row (Sequence[str | None]): Linha higienizada de dados da tabela.

        Returns:
            str | None: O número/código do Lote encontrado ou None.
        """
        # Captura "LOTE", "LOTE N°", "GRUPO 1", etc., ignorando espaços extras
        regex_lote = r"^\s*(?:LOTE|GRUPO)\b(?:\s*N[º°]?)?\s*[:|-]?\s*(\d+)"

        # Procura célula por célula
        for cell in row:
            if not cell:
                continue

            cell_upper = cell.upper().strip()

            # Se a célula tiver palavras de endereço, aborta a busca nesta célula
            if any(x in cell_upper for x in self.LOTE_BLOCK_WORDS):
                continue

            if "LOTE" in cell_upper or "GRUPO" in cell_upper:
                match_cell = re.search(regex_lote, cell_upper)
                if match_cell:
                    return match_cell.group(1)

        # Tenta encontrar na linha inteira concatenada
        # Usa compreensão de lista para garantir que 'None' não seja concatenado
        full_text = " ".join(str(c) for c in row if c).upper()

        if not any(x in full_text for x in self.LOTE_BLOCK_WORDS):
            match = re.search(regex_lote, full_text)

            if match:
                return match.group(1)

        return None

    def _update_lote_state(
        self, row: Sequence[str | None], idx_lote: int | None, state: ExtractionState
    ) -> str | None:
        """
        Extrai o lote da coluna específica e atualiza a memória de lote global (state).

        Args:
            row (Sequence[str | None]): Linha de dados atual.
            idx_lote (int | None): Índice da coluna de lote/grupo.
            state (ExtractionState): Estado global da extração.

        Returns:
            str | None: O lote normalizado encontrado ou o lote salvo no estado.
        """
        lote_raw = self._get_text_safe(row, idx_lote)

        if lote_raw:
            lote_match = re.search(r"\d+", lote_raw)

            if lote_match:
                state["last_lote"] = self._normalize_lote(lote_match.group())

            else:
                state["last_lote"] = lote_raw

        return state["last_lote"]

    def _normalize_lote(self, val: str | int | None) -> str | None:
        """
        Remove zeros à esquerda de lotes puramente numéricos.

        Args:
            val (str | int | None): Valor bruto do lote.

        Returns:
            str | None: Lote formatado.
        """
        if not val:
            return None

        s = str(val).strip()

        if s.isdigit():
            return str(int(s))

        return s

    def _update_item_counter(
        self, row: Sequence[str | None], idx_item: int | None, state: ExtractionState
    ) -> int:
        """
        Atualiza e retorna o contador sequencial de itens.
        Implementa proteção contra saltos discrepantes causados por lixo numérico.

        Args:
            row (Sequence[str | None]): Linha de dados atual.
            idx_item (int | None): Índice mapeado da coluna de itens.
            state (ExtractionState): Dicionário de estado contendo o 'item_counter'.

        Returns:
            int: O número consolidado do item atual.
        """
        item_str = self._get_text_safe(row, idx_item)
        item_raw = self._clean_number(item_str)

        if item_raw and item_raw > 0:
            final_item = int(item_raw)

            # Heurística de salto: Se o ID saltou mais de 10 posições de uma vez, assume que o
            # extrator leu um código solto em vez do ID do item
            if final_item > state["item_counter"] + 10:
                final_item = state["item_counter"]

            state["item_counter"] = final_item + 1

        # Caso não ache um número válido, incrementa a partir da memória
        else:
            final_item = state["item_counter"]
            state["item_counter"] += 1

        return final_item

    def _recover_item_from_row(
        self, row: Sequence[str | None], state: ExtractionState, current_lote: str | None
    ) -> ItemLicitacao | None:
        """
        Tenta pescar dados que escorregaram das colunas originais.
        Atua como fallback quando a formatação da tabela está corrompida.

        Args:
            row (Sequence[str | None]): Linha de dados atual.
            state (ExtractionState): Estado global da extração.
            current_lote (str | None): Lote detectado na linha atual, se houver.

        Returns:
            ItemLicitacao | None: Item reconstruído ou None se não conseguir recuperar dados
            válidos.
        """
        try:
            clean_row = [str(val).replace("\n", " ").strip() if val else "" for val in row]

            # Resgate da Descrição
            desc = self._recover_descricao(clean_row)
            if not desc:
                return None

            # Filtra todas as células que contêm números viáveis
            possible_numbers = []
            for cell in clean_row:
                num = self._clean_number(cell)

                if num is not None:
                    possible_numbers.append((num, cell))

            if not possible_numbers:
                return None

            # Resgate da Quantidade e Unidade de Fornecimento
            qtd, unid = self._recover_quantidade_and_unidade_fornecimento(
                possible_numbers, state["item_counter"]
            )

            if qtd is None:
                return None

            # Consolidação do Item e Estado
            final_item = state["item_counter"]
            state["item_counter"] += 1

            # Busca secundária por uma unidade genérica se a principal falhou
            if unid == "Unidade":
                for cell in clean_row:
                    if 0 < len(cell) <= self.MAX_UNID_LEN and cell.isalpha():
                        unid = cell.upper()
                        break

            final_lote = current_lote if current_lote else state.get("last_lote")

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
                f"Falha na heurística de resgate (Fallback). Linha ({row_sample}): {e}",
                exc_info=True,
            )
            return None

    def _recover_descricao(self, clean_row: list[str]) -> str | None:
        """
        Encontra a descrição assumindo que é o texto mais longo da linha que não é classificado
        como um número.

        Args:
            clean_row (list[str]): Linha higienizada de dados da tabela.

        Returns:
            str | None: Descrição recuperada ou None se não encontrar uma descrição válida.
        """
        candidates_desc = []

        for cell in clean_row:
            # A descrição deve ter um comprimento mínimo e não deve ser confundida com um número
            if len(cell) > self.MIN_DESC_LEN and not self._clean_number(cell):
                candidates_desc.append((len(cell), cell))

        if not candidates_desc:
            return None

        candidates_desc.sort(key=lambda x: x[0], reverse=True)

        return candidates_desc[0][1]

    def _recover_quantidade_and_unidade_fornecimento(
        self, possible_numbers: list[tuple[float, str]], current_item_idx: int
    ) -> tuple[int | None, str]:
        """
        Analisa números encontrados na linha para deduzir a quantidade e unidade.

        Args:
            possible_numbers (list[tuple[float, str]]): Lista de tuplas contendo números encontrados
            e seus textos originais.
            current_item_idx (int): O número do item atual para evitar confusão com a quantidade.

        Returns:
            tuple[int | None, str]: Quantidade numérica encontrada (ou None) e a unidade de
            fornecimento deduzida (ou "Unidade" como padrão).
        """
        qtd = None
        unid = "Unidade"

        for num, original_cell in possible_numbers:
            val = int(num)

            # Pula o ID do item atual/anterior e números gigantes
            if val in (current_item_idx, current_item_idx - 1) or val > self.MAX_VALID_QUANTITY:
                continue

            qtd = val
            text_in_cell = re.sub(r"[\d\.,]", "", original_cell).strip()

            if text_in_cell and len(text_in_cell) <= self.MAX_UNID_LEN:
                unid = text_in_cell.upper()

            break

        # Se não encontrou nas regras acima, pega o último número disponível
        if qtd is None and possible_numbers:
            last_num, last_cell = possible_numbers[-1]
            qtd = int(last_num)
            text_in_cell = re.sub(r"[\d\.,]", "", last_cell).strip()

            if text_in_cell and len(text_in_cell) <= self.MAX_UNID_LEN:
                unid = text_in_cell.upper()

        return qtd, unid

    def _deduplicate_items(self, items: list[ItemLicitacao]) -> list[ItemLicitacao]:
        """
        Agrupa itens extraídos pelo número do Lote e Item, mesclando dados de linhas
        espelhadas/repetidas para formar o item mais completo possível.

        Args:
            items (list[ItemLicitacao]): Lista bruta de itens extraídos.

        Returns:
            list[ItemLicitacao]: Lista de itens deduplicados, fundidos e ordenados.
        """
        if not items:
            return []

        final_map = {}

        for item in items:
            if self._is_garbage_item(item):
                continue

            lote = self._normalize_lote(item.lote)

            id_key = (lote, item.item)

            if id_key not in final_map:
                final_map[id_key] = item
            else:
                self._merge_duplicate_items(final_map[id_key], item)

        # Ordena a lista final priorizando o Lote, depois o ID do item
        return sorted(
            final_map.values(),
            key=lambda x: (
                int(x.lote) if x.lote and str(x.lote).isdigit() else 999,
                x.item if x.item is not None else 99999,
            ),
        )

    def _is_garbage_item(self, item: ItemLicitacao) -> bool:
        """
        Verifica se o item é um falso positivo baseado no tamanho e conteúdo da descrição.

        Args:
            item (ItemLicitacao): O item a ser avaliado.

        Returns:
            bool: True se o item for considerado lixo, False caso contrário.
        """
        if not item.objeto or len(item.objeto) < self.MIN_DESC_LEN:
            return True

        # Se a descrição for composta APENAS por caracteres inúteis, é lixo
        return set(item.objeto.strip().lower()) <= self.GARBAGE_CHARS

    def _merge_duplicate_items(self, existing: ItemLicitacao, new_item: ItemLicitacao) -> None:
        """
        Mescla dois itens de mesmo ID, priorizando a informação mais rica.
        Aplica regras de negócio para salvar a maior descrição e a melhor unidade.

        Args:
            existing (ItemLicitacao): O item já existente no mapa de resultados.
            new_item (ItemLicitacao): O novo item a ser comparado e possivelmente mesclado.
        """
        # Prioriza a descrição mais longa, assumindo que é mais detalhada e informativa
        if len(new_item.objeto) > len(existing.objeto):
            existing.objeto = new_item.objeto

        # Prioriza quantidades reais, protegendo contra None ou 0 da primeira leitura
        if (not existing.quantidade or existing.quantidade <= 1) and (
            new_item.quantidade and new_item.quantidade > 1
        ):
            existing.quantidade = new_item.quantidade

        # Refina a Unidade de Fornecimento
        new_u = str(new_item.unidade_fornecimento).strip().upper()
        old_u = str(existing.unidade_fornecimento).strip().upper()

        # Prioriza Unidades de Fornecimento mais específicas e detalhadas
        if (old_u in self.GENERIC_UNITS and new_u not in self.GENERIC_UNITS) or (
            old_u not in self.GENERIC_UNITS
            and new_u not in self.GENERIC_UNITS
            and len(new_item.unidade_fornecimento) > len(existing.unidade_fornecimento)
        ):
            existing.unidade_fornecimento = new_item.unidade_fornecimento

        if new_item.lote and not existing.lote:
            existing.lote = new_item.lote

    def _clean_unidade_fornecimento(self, val: str | None) -> str:
        """
        Higieniza a unidade de fornecimento.
        Esta função remove números e pontuações iniciais para isolar apenas o texto da unidade.

        Args:
            val (str | None): Valor bruto extraído da coluna de unidade.

        Returns:
            str: Unidade de fornecimento limpa (ou "Unidade" como padrão caso inválida).
        """
        if not val:
            return "Unidade"

        # Remove dígitos, pontos, vírgulas e espaços do início da string
        cleaned = re.sub(r"^[\d\.,\s]+", "", str(val)).strip()

        # Se após a limpeza não sobrar nada, ou sobrar um texto gigante, assumime o valor padrão
        if not cleaned or len(cleaned) > self.MAX_UNID_LEN:
            return "Unidade"

        return cleaned

    @classmethod
    def _clean_number(cls, val: str | None) -> float | None:
        """
        Extrai o primeiro número válido de uma string ruidosa.
        Aborta se a string contiver excesso de letras, indicando ser um texto comum.

        Args:
            val (str | None): String ruidosa extraída de uma célula.

        Returns:
            float | None: Número decimal/inteiro ou None se a string for inválida.
        """
        if not val:
            return None

        try:
            clean = val.strip()

            # Impede a extração de números dentro de textos puramente descritivos
            if len(re.findall(r"[a-zA-Z]", clean)) > cls.MAX_ALPHA_IN_NUMBER:
                return None

            clean = clean.replace(".", "").replace(",", ".")
            match = re.search(r"[\d\.]+", clean)

            if match:
                num = float(match.group())
                return num if num > 0 else None

            return None

        except Exception:
            return None
