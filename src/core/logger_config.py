import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """
    Configura o sistema de logging para a aplicação.

    Args:
        verbose (bool): Se True, define o nível de log para DEBUG, caso contrário INFO
    """

    level = logging.DEBUG if verbose else logging.INFO

    # Formato mais limpo para CLI
    formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")

    # Configura o handler para saída padrão (console)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configura o logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Limpa handlers antigos
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(handler)

    # Reduz barulho de bibliotecas externas
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
