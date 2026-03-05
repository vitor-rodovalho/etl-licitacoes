import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import pdfplumber
from pdfplumber.pdf import PDF

from src.extractors.base import BaseExtractor, ExtractionState
from src.schemas.licitacao import ItemLicitacao

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """
    Extrator Híbrido com Modo Especialista para 'Relação de Itens'.
    """

    # =========================================================================
    # CONSTANTES ESPECÍFICAS DE PDF
    # =========================================================================

    # Padrão de campos intrusos que grudam nos valores no formato "Relação de Itens"
    RELACAO_ITENS_INTRUDER_PATTERN = re.compile(
        r"critério de|menor preço|valor estimado|valor unitário|valor total|"
        r"unidade de fornecimento|quantidade m[íi]nima|quantidade m[áa]xima|"
        r"intervalo m[íi]nimo|local de entrega|grupo:",
        re.IGNORECASE,
    )

    # Regex para capturar DD/MM/AAAA HH:MM (padrão de rodapé de sistema)
    FOOTER_DATE_PATTERN = re.compile(r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}")

    # Termos financeiros ou de editais que o parser de texto deve pular imediatamente
    RELACAO_ITENS_STOP_KEYS = frozenset(
        [
            "tratamento diferenciado",
            "aplicabilidade decreto",
            "quantidade mínima",
            "critério de julgamento",
            "menor preço",
            "critério de valor",
            "valor estimado",
            "valor unitário",
            "quantidade máxima",
            "intervalo mínimo",
            "local de entrega",
            "valor total",
        ]
    )

    # Palavras que indicam cabeçalhos/rodapés inúteis no meio da página
    RELACAO_ITENS_RODAPE_WORDS = frozenset(["página", "uasg", "fls."])

    # Palavras que acionam o "Kill Switch" se encontradas dentro da coluna de descrição
    GARBAGE_DESC_WORDS = frozenset(
        [
            "ANEXO",
            "TERMO DE",
            "DECLARAÇÃO",
            "EDITAL",
            "PREGÃO",
            "RODAPÉ",
            "CONSEQUÊNCIA",
            "RISCO",
            "SANÇÕES",
        ]
    )

    # Palavras e trechos que indicam que a linha é um rodapé, cabeçalho ou metadado do PDF
    TRASH_KEYWORDS = frozenset(
        [
            "cnpj:",
            "cep:",
            "telefone:",
            "tel:",
            "endereço:",
            "e-mail:",
            "email:",
            "valor total estimado",
            "documento assinado",
            "código verificador",
            "autenticidade do",
            "eproc.",
            "local e data",
            "valor máximo",
        ]
    )

    # Termos sem espaço que indicam linhas de totalizadores financeiros que devem ser ignoradas
    FINANCIAL_TOTALS_KEYS = frozenset(
        [
            "valortotal",
            "valorglobal",
            "valormáximo",
            "xxxx",
        ]
    )

    # Limites para detecção de fallback do número do item
    MAX_ITEM_NUM_DIGITS = 4
    MAX_RAW_ITEM_LENGTH = 7

    # Comprimento máximo aceitável para uma descrição
    MAX_DESC_LEN = 1500

    # Tamanho mínimo da descrição se ela parecer ser apenas um número
    MIN_DESC_NUMBER_LEN = 15

    # Tamanho máximo de uma string de quantidade
    MAX_QTD_STRING_LEN = 25

    # Tamanho máximo de uma célula para ela não ser mascarada ("cegada") no resgate lateral
    MAX_QTD_CELL_LEN = 20

    # Quantidade de partes esperadas ao dividir um número decimal/financeiro pela vírgula
    EXPECTED_DECIMAL_PARTS = 2

    # =========================================================================
    # EXTRAÇÃO E MÉTODOS AUXILIARES
    # =========================================================================

    def extract(self, file_path: Path) -> list[ItemLicitacao]:
        """
        Extrai itens de um arquivo PDF, aplicando uma estratégia híbrida que inclui um modo
        especialista para documentos do tipo 'Relação de Itens', uma abordagem tradicional de
        extração de tabelas e um fallback para casos problemáticos.

        Args:
            file_path (Path): Caminho para o arquivo PDF a ser processado.

        Returns:
            list[ItemLicitacao]: Lista de itens extraídos do PDF.
        """
        if not file_path.exists() or not file_path.is_file():
            logger.error(f"Arquivo não encontrado ou inválido: {file_path}")
            return []

        extracted = []
        try:
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"Iniciando PDF: {file_path.name} ({len(pdf.pages)} págs)")

                # Obtém o texto completo para análise
                full_text = ""
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        full_text += t + "\n"

                # Verifica se o documento é uma 'Relação de Itens' antes de tentar extrair tabelas
                if self._is_relacao_itens(full_text):
                    logger.info("Detectado formato 'Relação de Itens'. Usando parser específico.")

                    # Se for, chama o parser específico das Relações de Itens
                    return self._parse_relacao_itens(full_text)

                # Tenta extrair usando a estratégia tradicional de tabelas
                logger.debug(f"[{file_path.name}] Extraindo tabelas.")
                extracted = self._extract_tables(pdf)

        except Exception as e:
            logger.error(f"Falha crítica ao ler o .PDF '{file_path.name}': {e}", exc_info=True)

        final_items = self._deduplicate_items(extracted)

        logger.info(f"Concluído PDF: {file_path.name}. Itens brutos extraídos: {len(final_items)}")

        return final_items

    def _is_relacao_itens(self, text: str) -> bool:
        """
        Verifica se o documento é uma Relação de Itens oficial.

        Busca por padrões específicos como "Relação de Itens", "Pregão" e "1 - itens da licitação"
        nos primeiros 1000 caracteres do documento.

        Args:
            text (str): Texto completo extraído do PDF.

        Returns:
            bool: True se o documento for identificado como Relação de Itens, False caso contrário.
        """

        head = text[:1000].lower()

        has_titulo = "relação de itens" in head
        has_pregao = "pregão" in head or "licitação" in head
        has_marcador_itens = "1 - itens da licitação" in head

        return has_titulo and has_pregao and has_marcador_itens

    def _parse_relacao_itens(self, full_text: str) -> list[ItemLicitacao]:
        """
        Parser para documentos Relações de Itens oficiais.
        Orquestra a leitura linha a linha delegando o processamento aos helpers.

        Args:
            full_text (str): Texto completo do PDF extraído.

        Returns:
            list[ItemLicitacao]: Lista de itens extraídos do documento.
        """
        lines = self._trim_relacao_itens_text(full_text)
        items = []
        curr_item = {}
        item_start_pattern = re.compile(r"^\s*(\d+)\s+-\s+(.+)$")

        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue

            # Detecta Início de um Novo Item
            match = item_start_pattern.match(line)
            if match:
                title = match.group(2).strip()
                if "itens da licitação" in title.lower():
                    continue

                self._save_item_relacao_itens(curr_item, items)
                curr_item = {"item": int(match.group(1)), "objeto": title, "full_desc": []}
                continue

            # Se não inicializou o primeiro item ainda, ignora
            if not curr_item:
                continue

            lower_line = line.lower()

            # Tenta processar como um campo chave
            if self._process_relacao_itens_fields(line, lower_line, lines, i, curr_item):
                continue

            # Se não for campo chave, é texto livre
            if "full_desc" in curr_item:
                if any(x in lower_line for x in self.RELACAO_ITENS_RODAPE_WORDS):
                    continue
                if self.FOOTER_DATE_PATTERN.search(line):
                    continue

                curr_item["full_desc"].append(line)

        # Salva o último item processado
        self._save_item_relacao_itens(curr_item, items)

        return self._deduplicate_items(items)

    def _trim_relacao_itens_text(self, full_text: str) -> list[str]:
        """
        Corta o cabeçalho e o rodapé das Relações de Itens e retorna as linhas limpas.

        Args:
            full_text (str): Texto completo do PDF extraído.

        Returns:
            list[str]: Linhas de texto relevantes para a extração dos itens.
        """
        marker_match = re.search(r"1\s*[-]\s*Itens\s+da\s+Licitação", full_text, re.IGNORECASE)
        if marker_match:
            full_text = full_text[marker_match.start() :]

        stop_match = re.search(r"2\s*[-]\s*Composição\s+dos\s+Grupos", full_text, re.IGNORECASE)
        if stop_match:
            full_text = full_text[: stop_match.start()]

        return full_text.split("\n")

    def _process_relacao_itens_fields(
        self, line: str, lower: str, lines: list[str], idx: int, curr_item: dict
    ) -> bool:
        """
        Identifica qual campo as linhas das Relações de Itens contêm e delega a extração.

        Args:
            line (str): Linha de texto atual.
            lower (str): Linha de texto em minúsculas para facilitar a detecção.
            lines (list[str]): Lista de todas as linhas do documento.
            idx (int): Índice da linha atual na lista de linhas.
            curr_item (dict): Dicionário do item atual sendo construído.

        Returns:
            bool: True se a linha foi processada como um campo chave, False caso contrário.
        """
        if "quantidade total:" in lower:
            self._extract_relacao_itens_quantidade(line, lower, lines, idx, curr_item)
            return True

        if "unidade de fornecimento:" in lower:
            self._extract_relacao_itens_unidade(line, lines, idx, curr_item)
            return True

        if "grupo:" in lower:
            self._extract_relacao_itens_grupo(line, lines, idx, curr_item)
            return True

        if "descrição detalhada:" in lower:
            self._extract_relacao_itens_desc(line, curr_item)
            return True

        return any(k in lower for k in self.RELACAO_ITENS_STOP_KEYS)

    def _extract_relacao_itens_quantidade(
        self, line: str, lower: str, lines: list[str], idx: int, curr_item: dict
    ) -> None:
        """
        Extrai a quantidade das Relações de Itens.

        Args:
            line (str): Linha de texto atual.
            lower (str): Linha de texto em minúsculas para facilitar a detecção.
            lines (list[str]): Lista de todas as linhas do documento.
            idx (int): Índice da linha atual na lista de linhas.
            curr_item (dict): Dicionário do item atual sendo construído.
        """
        clean_quantidade = self._clean_number(self._get_value_relacao_itens(line, lines, idx))
        if clean_quantidade:
            curr_item["quantidade"] = int(clean_quantidade)

    def _extract_relacao_itens_unidade(
        self, line: str, lines: list[str], idx: int, curr_item: dict
    ) -> None:
        """
        Extrai e limpa a unidade de fornecimento das Relações de Itens.

        Args:
            line (str): Linha de texto atual.
            lines (list[str]): Lista de todas as linhas do documento.
            idx (int): Índice da linha atual na lista de linhas.
            curr_item (dict): Dicionário do item atual sendo construído.
        """
        val = self._get_value_relacao_itens(line, lines, idx)
        if not val or val.strip().replace(".", "").replace(",", "").isdigit():
            return

        val = re.sub(r"^[\d\.,\s]+", "", val)

        curr_item["unidade_fornecimento"] = val.strip()

    def _extract_relacao_itens_grupo(
        self, line: str, lines: list[str], idx: int, curr_item: dict
    ) -> None:
        """
        Extrai o lote/grupo das Relações de Itens.

        Args:
            line (str): Linha de texto atual.
            lines (list[str]): Lista de todas as linhas do documento.
            idx (int): Índice da linha atual na lista de linhas.
            curr_item (dict): Dicionário do item atual sendo construído.
        """
        val = self._get_value_relacao_itens(line, lines, idx)
        if val:
            curr_item["lote"] = re.sub(r"(?i)grupo\s*:?", "", val).strip()

    def _extract_relacao_itens_desc(self, line: str, curr_item: dict) -> None:
        """
        Extrai a descrição dos itens das Relações de Itens.

        Args:
            line (str): Linha de texto atual.
            curr_item (dict): Dicionário do item atual sendo construído.
        """
        parts = line.split(":", 1)
        if len(parts) > 1 and parts[1].strip():
            curr_item["full_desc"].append(parts[1].strip())

    def _get_value_relacao_itens(self, line: str, lines: list, idx: int) -> str:
        """
        Helper que fatia valores grudados gerados pela extração de colunas do PDF.

        Args:
            line (str): Linha de texto atual.
            lines (list[str]): Lista de todas as linhas do documento.
            idx (int): Índice da linha atual na lista de linhas.

        Returns:
            str: Valor extraído ou string vazia se não encontrar nada válido.
        """
        parts = line.split(":", 1)

        if len(parts) > 1:
            val_inline = parts[1].strip()
            val_inline = self.RELACAO_ITENS_INTRUDER_PATTERN.split(val_inline)[0].strip()

            # Só retorna se sobrar um valor real. Se sobrou vazio, a busca continua
            if val_inline and val_inline != ":":
                return val_inline

        # Se o valor não está na mesma linha, desce procurando
        for offset in range(1, 9):
            if idx + offset < len(lines):
                candidate = lines[idx + offset].strip()
                if not candidate:
                    continue

                clean_candidate = self.RELACAO_ITENS_INTRUDER_PATTERN.split(candidate)[0].strip()

                if clean_candidate and clean_candidate != ":":
                    # Se após limpar, ainda tem um ":" e não é uma data/hora, é outro campo
                    if ":" in clean_candidate and not self.FOOTER_DATE_PATTERN.search(candidate):
                        continue
                    return clean_candidate

        return ""

    def _save_item_relacao_itens(self, curr: dict, items: list) -> None:
        """
        Salva o item atual das Relações de Itens na lista final de itens.

        Args:
            curr (dict): Dicionário do item atual sendo construído.
            items (list): Lista de itens extraídos até agora, onde o item finalizado é adicionado.

        """
        if "item" in curr:
            body_desc = " ".join(curr["full_desc"]).strip()

            if not body_desc:
                return

            items.append(
                ItemLicitacao(
                    item=curr["item"],
                    objeto=body_desc,
                    quantidade=curr.get("quantidade", 1),
                    unidade_fornecimento=curr.get("unidade_fornecimento", "Unidade"),
                    lote=curr.get("lote"),
                )
            )
        curr.clear()

    def _extract_tables(self, pdf: PDF) -> list[ItemLicitacao]:
        """
        Lê e extrai dados das tabelas do documento PDF fornecido.

        Args:
            pdf (PDF): Objeto PDFplumber já aberto e carregado.

        Returns:
            list[ItemLicitacao]: Lista de itens extraídos usando a estratégia tradicional.
        """
        items = []
        state: ExtractionState = {
            "last_lote": None,
            "item_counter": 1,
            "current_header_map": None,
            "pending_broken_desc": None,
            "last_extracted_item": None,
            "pending_item_num": None,
        }

        for page in pdf.pages:
            tables = page.extract_tables()

            # Se a página não tem linhas de tabela, apenas pula
            if not tables:
                continue

            for table in tables:
                if table:
                    items.extend(self._process_table_rows(table, state, parse_func=self._parse_row))

        return items

    def _process_table_rows(
        self,
        rows: list[list[str | None]],
        state: ExtractionState,
        parse_func: Callable[
            [Sequence[str | None], dict[str, int], ExtractionState], ItemLicitacao | None
        ],
    ) -> list[ItemLicitacao]:
        """
        Processa as linhas de uma tabela extraída do PDF, identificando lotes, cabeçalhos e itens.

        Aplica heurísticas para detectar declarações de lote, mapeamento de colunas e extração de
        itens individuais, mantendo estado entre linhas para tratamento de campos distribuídos
        em múltiplas linhas.

        Args:
            rows (list[list[str | None]]): Linhas da tabela extraída do PDF.
            state (ExtractionState): Estado compartilhado entre linhas da tabela.
            parse_func (Callable): Função de parsing a ser aplicada para extrair os itens de cada
            linha.

        Returns:
            list[ItemLicitacao]: Lista de itens extraídos da tabela.
        """
        items = []
        current_lote = None
        clean_rows = self._clean_rows(rows)

        for row in clean_rows:
            # Se a linha for um lote, atualiza o estado e pula para a próxima linha
            lote = self._extract_lote_from_row(row)
            if lote:
                norm_lote = self._normalize_lote(lote)
                current_lote = norm_lote
                state["last_lote"] = norm_lote
                continue

            # Se a linha for um cabeçalho, atualiza o estado e pula para a próxima linha
            header = self._identify_columns(row)

            if header and "quantidade" in header and "objeto" in header:
                state["current_header_map"] = header
                continue

            # Se a linha não for lote nem cabeçalho, tenta extrair um item com a função de parsing
            if state["current_header_map"]:
                item = self._create_item_from_row(
                    row, state["current_header_map"], state, current_lote, parse_func
                )
                if item:
                    items.append(item)

        return items

    def _create_item_from_row(
        self,
        row: Sequence[str | None],
        header_map: dict[str, int],
        state: ExtractionState,
        current_lote: str | None,
        parse_func: Callable[
            [Sequence[str | None], dict[str, int], ExtractionState], ItemLicitacao | None
        ],
    ) -> ItemLicitacao | None:
        """
        Cria um objeto ItemLicitacao a partir de uma linha de tabela usando a função de parsing
        fornecida.

        Args:
            row (Sequence[str | None]): Linha de tabela a ser processada.
            header_map (dict[str, int]): Mapeamento de colunas identificado para a tabela.
            state (ExtractionState): Estado compartilhado para controle de lotes e itens pendentes.
            current_lote (str | None): Lote atual identificado para associar ao item.
            parse_func (Callable): Função de parsing a ser aplicada para extrair o item da linha.

        Returns:
            ItemLicitacao | None: Objeto ItemLicitacao extraído ou None se a linha for considerada
            inválida ou lixo.
        """
        row_str = " ".join([str(c) for c in row if c]).lower()
        row_str_no_space = row_str.replace(" ", "")

        # Vasculha a linha atual para ver se ela anuncia um número de item
        for cell in row:
            if cell:
                # Captura padrões como "ITEM 01", "Item: 2", "ITEM N° 10" no início da célula
                match = re.search(r"(?i)^ITEM\s*(?:N[°º]\s*|:\s*)?(\d+)", str(cell).strip())
                if match:
                    state["pending_item_num"] = int(match.group(1))
                    break

        # Verifica se a linha contém lixo (termos financeiros ou rodapés/sensíveis)
        is_financial_trash = any(key in row_str_no_space for key in self.FINANCIAL_TOTALS_KEYS)
        is_general_trash = any(kw in row_str for kw in self.TRASH_KEYWORDS)

        if is_financial_trash or is_general_trash:
            state["pending_broken_desc"] = None
            state["last_extracted_item"] = None
            return None

        # Determina o lote ativo para associar ao item
        active = current_lote if current_lote else state.get("last_lote")
        if active:
            active = self._normalize_lote(active)
        state["last_lote"] = active

        item = parse_func(row, header_map, state)

        # Regras de limpeza final para a unidade de fornecimento e descrição
        if item:
            unid_str = str(item.unidade_fornecimento).strip().upper()

            if (
                not any(c.isalpha() for c in unid_str)
                or set(unid_str) == {"X"}
                or item.objeto.lower().strip().startswith(self.INVALID_DESC_PREFIXES)
                or len(item.objeto) < self.MIN_DESC_LEN
            ):
                return None

            if not item.lote and active:
                item.lote = active

        return item

    def _parse_row(
        self, row: Sequence[str | None], mapping: dict[str, int], state: ExtractionState
    ) -> ItemLicitacao | None:
        """
        Realiza o parsing de uma linha de tabela usando um mapeamento de colunas específico.

        Args:
            row: Linha de tabela a ser processada.
            mapping: Mapeamento de colunas identificadas para a tabela.
            state: Estado compartilhado para controle de lotes e itens pendentes.

        Returns:
            ItemLicitacao | None: Objeto ItemLicitacao extraído ou None se a linha for considerada
            inválida ou lixo.
        """
        try:
            if not mapping:
                return None

            # Extrações Iniciais
            raw_desc = self._get_text_safe(row, mapping.get("objeto"))
            val_qtd = self._get_text_safe(row, mapping.get("quantidade"))

            raw_item = self._clean_number(self._get_text_safe(row, mapping.get("item")))
            try:
                item_num = int(float(raw_item)) if raw_item is not None else None
            except (ValueError, TypeError):
                item_num = None

            # Se o mapa falhou, mas a primeira coluna é um número isolado, é o Item
            if item_num is None:
                cand_item = str(self._get_text_safe(row, 0) or "").strip()
                clean_cand_item = re.sub(r"[^\d]", "", cand_item)
                if (
                    clean_cand_item
                    and len(clean_cand_item) <= self.MAX_ITEM_NUM_DIGITS
                    and len(cand_item) <= self.MAX_RAW_ITEM_LENGTH
                ):
                    item_num = int(clean_cand_item)

            qtd = self._extract_quantidade_table(row, val_qtd, mapping.get("quantidade"))
            has_valid_desc = bool(raw_desc and len(raw_desc) >= self.MIN_DESC_LEN)

            # Se a descrição tem palavras que indicam lixo, ignora a linha inteira e reseta buffers
            if has_valid_desc and any(w in raw_desc.upper() for w in self.GARBAGE_DESC_WORDS):
                state["last_extracted_item"] = None
                state["pending_broken_desc"] = None
                return None

            # Resolve a fragmentação de linhas
            final_desc = self._resolve_row_buffer(state, has_valid_desc, qtd, item_num, raw_desc)

            if not qtd or not final_desc or len(final_desc) < self.MIN_DESC_LEN:
                return None

            # Remove lixo de "R$" vazio que possa ter vazado das colunas de valores
            final_desc = re.sub(r"(?i)\bR\$\s*R\$?\b", "", final_desc).strip()
            final_desc = re.sub(r"(?i)\bR\$\b", "", final_desc).strip()

            if (
                self._clean_number(final_desc) is not None
                and len(final_desc) < self.MIN_DESC_NUMBER_LEN
            ):
                return None

            # Criação do Objeto
            final_item = self._update_item_counter(row, mapping.get("item"), state)
            final_lote = self._update_lote_state(row, mapping.get("lote"), state)
            unid = self._extract_unidade_table(row, val_qtd, mapping.get("unidade_fornecimento"))

            # Se usar o item flutuante, força-o como o número oficial e sincroniza o contador
            pending_num = state.get("pending_item_num")
            if pending_num is not None:
                safe_pending_num = int(pending_num)
                final_item = safe_pending_num
                state["item_counter"] = safe_pending_num + 1
                state["pending_item_num"] = None

            novo_item = ItemLicitacao(
                item=final_item,
                quantidade=qtd,
                objeto=final_desc,
                unidade_fornecimento=unid,
                lote=final_lote,
            )

            # Guarda referência para caso a próxima linha seja de continuação
            state["last_extracted_item"] = novo_item

            return novo_item

        except Exception as e:
            row_sample = str(row)[:60] + "..." if row else "Linha Vazia"
            logger.debug(
                f"Falha no parser de tabelas (PDF). Linha ({row_sample}): {e}", exc_info=True
            )
            return None

    def _resolve_row_buffer(
        self,
        state: ExtractionState,
        has_valid_desc: bool,
        qtd: int | None,
        item_num: int | None,
        raw_desc: str,
    ) -> str | None:
        """
        Gerencia o buffer para costurar linhas que foram quebradas no PDF.

        Args:
            state (ExtractionState): Dicionário que guarda o estado atual da extração.
            has_valid_desc (bool): Indicador que confirma se a linha atual possui um
            texto de descrição válido.
            qtd (int | None): A quantidade numérica do produto encontrada na linha atual.
            item_num (int | None): Número do item extraído da linha atual.
            raw_desc (str): O texto bruto da coluna de descrição extraído da linha atual.

        Returns:
            str | None: Retorna a string da descrição completa se as condições indicarem
            que o item está finalizado ou None caso contrário.
        """
        pending_desc = state.get("pending_broken_desc")
        final_desc = None

        # Se a descrição é válida, mas não tem quantidade nem número de item, pode ser uma
        # linha de continuação da descrição
        if has_valid_desc and not qtd and not item_num:
            last_item = state.get("last_extracted_item")
            if last_item and len(last_item.objeto) < self.MAX_DESC_LEN:
                last_item.objeto += " " + raw_desc

        # Se a descrição é válida e tem número de item, mas não tem quantidade, pode ser início
        # de uma descrição quebrada. Guarda no buffer para tentar completar na próxima linha
        elif has_valid_desc and not qtd and item_num:
            state["pending_broken_desc"] = raw_desc
            state["last_extracted_item"] = None

        # Se a descrição não é válida, mas tem quantidade, e tem uma descrição pendente no
        # buffer, é o fim da descrição quebrada. Junta e limpa o buffer
        elif not has_valid_desc and qtd and pending_desc:
            final_desc = pending_desc
            state["pending_broken_desc"] = None

        # Se a descrição é válida e tem quantidade, é um caso normal
        elif has_valid_desc and qtd:
            final_desc = (pending_desc + " " + raw_desc) if pending_desc else raw_desc
            state["pending_broken_desc"] = None

        return final_desc

    def _extract_quantidade_table(
        self, row: Sequence[str | None], val_qtd: str, idx_qtd: int | None
    ) -> int | None:
        """
        Limpa a string de quantidade e tenta resgate lateral se falhar.

        Args:
            row (Sequence[str | None]): Linha de dados atual.
            val_qtd (str): Valor bruto extraído da coluna de quantidade.
            idx_qtd (int | None): Índice da coluna de quantidade.

        Returns:
            int | None: Quantidade numérica encontrada ou None.
        """
        qtd = None

        if val_qtd:
            val_limpo = val_qtd.strip().upper()

            # Trata valores financeiros
            if not (val_limpo.startswith("R$") or val_limpo.startswith("RS")):
                val_sem_moeda = re.sub(r"R\$\s*[\d\.,]+", "", val_limpo)
                val_sem_moeda = re.sub(r",\d{2}\b", "", val_sem_moeda)

                if len(val_sem_moeda) < self.MAX_QTD_STRING_LEN:
                    nums = re.findall(r"\d+", val_sem_moeda.replace(".", ""))
                    if nums:
                        qtd = int(nums[0])

        if not qtd:
            safe_row = []
            for c in row:
                cstr = str(c).strip() if c else ""
                # Mascara células perigosas (datas, dinheiros, anexos) para a heurística
                if (
                    len(cstr) > self.MAX_QTD_CELL_LEN
                    or re.search(r"\d{2}/\d{2}/\d{4}", cstr)
                    or "ANEXO" in cstr.upper()
                    or ("R$" in cstr.upper())
                    or (
                        "," in cstr
                        and len(cstr.split(",")) == self.EXPECTED_DECIMAL_PARTS
                        and cstr.split(",")[1][:2].isdigit()
                    )
                ):
                    safe_row.append("XXX")
                else:
                    safe_row.append(c)
            qtd = self._extract_quantidade_heuristic(safe_row, idx_qtd)

        return qtd

    def _extract_quantidade_heuristic(
        self, row: Sequence[str | None], idx_qtd: int | None
    ) -> int | None:
        """
        Tenta encontrar a quantidade nas colunas vizinhas caso a coluna mapeada falhe ou tenha
        sofrido deslocamento.

        Args:
            row (Sequence[str | None]): Linha de dados atual.
            idx_qtd (int | None): Índice original esperado para a quantidade.

        Returns:
            int | None: Quantidade numérica encontrada ou None.
        """
        if idx_qtd is None:
            return None

        # Ordem de tentativa: própria coluna, próxima, anterior
        indices_to_try = [idx_qtd, idx_qtd + 1, idx_qtd - 1]

        for i in indices_to_try:
            if 0 <= i < len(row):
                candidate_str = self._get_text_safe(row, i)
                if not candidate_str:
                    continue

                cand_strip = candidate_str.strip()
                clean_cand_qtd = re.sub(r"[^\d]", "", cand_strip)

                if (
                    i == 0
                    and idx_qtd != 0
                    and clean_cand_qtd
                    and len(clean_cand_qtd) <= self.MAX_ITEM_NUM_DIGITS
                    and len(cand_strip) <= self.MAX_RAW_ITEM_LENGTH
                ):
                    continue

                qtd_candidate = self._clean_number(candidate_str)

                if qtd_candidate:
                    return int(qtd_candidate)

        return None

    def _extract_unidade_table(
        self, row: Sequence[str | None], val_qtd: str, idx_unid: int | None
    ) -> str:
        """
        Extrai a unidade e aplica a regra de resgate da coluna de unidade de fornecimento.

        Args:
            row (Sequence[str | None]): Linha de dados atual.
            val_qtd (str): Valor bruto extraído da coluna de quantidade.
            idx_unid (int | None): Índice da coluna de unidade de fornecimento.

        Returns:
            str: Unidade de fornecimento extraída ou limpa.
        """
        unid_raw = None

        if idx_unid is not None:
            # Ordem de tentativa: própria coluna, próxima, anterior
            indices_to_try = [idx_unid, idx_unid + 1, idx_unid - 1]

            for i in indices_to_try:
                candidate = self._get_text_safe(row, i)

                # Verifica se o candidato tem texto e contém letras
                if candidate and any(c.isalpha() for c in candidate):
                    unid_raw = candidate
                    break

        unid = self._clean_unidade_fornecimento(unid_raw)

        if unid.upper() == "UNIDADE" and val_qtd:
            possible_unid = re.sub(r"[\d\.,]", "", val_qtd)
            possible_unid = re.sub(r"(?i)R\$|RS", "", possible_unid).strip()
            if (
                possible_unid
                and len(possible_unid) <= self.MAX_UNID_LEN
                and any(c.isalpha() for c in possible_unid)
            ):
                unid = self._clean_unidade_fornecimento(possible_unid)

        return unid
