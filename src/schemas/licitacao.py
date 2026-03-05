from pydantic import BaseModel, Field, field_validator


class ItemLicitacao(BaseModel):
    """
    Representa um item individual extraído dos anexos.
    """

    lote: str | None = Field(
        default=None,
        description="Identificador do lote/grupo (ex: 'G1', '1'). Null se não houver.",
    )
    item: int = Field(
        ..., gt=0, description="Número sequencial do item. Deve ser maior que 0."
    )
    objeto: str = Field(
        ..., min_length=3, description="Descrição completa do item licitado."
    )
    quantidade: int = Field(
        ..., gt=0, description="Quantidade solicitada. Deve ser um inteiro positivo."
    )
    unidade_fornecimento: str = Field(
        default="Unidade",  # Valor default seguro caso a extração falhe
        description="Unidade de medida (ex: 'UN', 'Caixa').",
    )

    @field_validator("lote", mode="before")
    @classmethod
    def clean_lote(cls, v: object) -> str | None:
        """
        Normaliza o lote para garantir que strings vazias virem None.

        Args:
            v (object): Valor original do lote.

        Returns:
            str | None: Lote normalizado ou None se for vazio.
        """

        if v is None:
            return None

        if isinstance(v, str):
            v_limpo = v.strip()

            if not v_limpo:
                return None

            return v_limpo

        # Se for qualquer outra coisa (ex: int 10), converte para string
        return str(v)


class Licitacao(BaseModel):
    """
    Modelo final que representa o objeto JSON de uma licitação processada.
    """

    arquivo_json: str = Field(..., description="Nome do arquivo JSON de origem.")
    numero_pregao: str = Field(
        ..., description="Número do pregão identificado nos metadados."
    )
    orgao: str = Field(..., description="Órgão público responsável.")
    cidade: str = Field(default="", description="Município do órgão licitante.")
    estado: str = Field(
        default="",
        pattern=r"^[A-Z]{2}$",
        description="Sigla da UF (2 caracteres maiúsculos).",
    )
    anexos_processados: list[str] = Field(
        default_factory=list,
        description="Lista dos nomes dos arquivos PDF/DOCX que foram lidos.",
    )
    itens_extraidos: list[ItemLicitacao] = Field(
        default_factory=list, description="Lista de itens encontrados nos anexos."
    )

    model_config = {
        "populate_by_name": True,
        # Remove espaços em branco de todas as strings automaticamente
        "str_strip_whitespace": True,
        "json_schema_extra": {
            "example": {
                "arquivo_json": "2024-conlicitacao-123.json",
                "numero_pregao": "PE 900/2024",
                "orgao": "Prefeitura Municipal de Goiânia",
                "cidade": "Goiânia",
                "estado": "GO",
                "anexos_processados": ["edital_retificado.pdf", "anexo_lotes.pdf"],
                "itens_extraidos": [
                    {
                        "lote": "1",
                        "item": 1,
                        "objeto": "Computador Desktop i7 16GB",
                        "quantidade": 10,
                        "unidade_fornecimento": "Unidade",
                    },
                    {
                        "lote": "1",
                        "item": 2,
                        "objeto": "Monitor 24 Polegadas",
                        "quantidade": 10,
                        "unidade_fornecimento": "Unidade",
                    },
                ],
            }
        },
    }
