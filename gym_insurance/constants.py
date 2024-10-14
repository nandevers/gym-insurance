from pathlib import Path
import logging

MODELSPATH = Path("./models")
DATAPATH = Path("./.temp/data")


BUDGET_LIST = [10_000_000] #, 5_000_000, 10_000_000, 15_000_000]
# BUDGET_LIST = [15_000_000 , 20_000_000, 25_000_000, 30_000_000]
#BUDGET_LIST = [ 1_000_000, 200_000, 300_000]
# BUDGET_LIST = [10_000]


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


configure_logging()
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

STD_COLS = [
    "valor_indenizacao",
    "nr_previous_proposals",
    "max_nr_area_total",
    "min_nr_area_total",
    "sum_nr_area_total",
    "max_niveldecobertura",
    "min_niveldecobertura",
    "sum_niveldecobertura",
    "max_valor_indenizacao",
    "min_valor_indenizacao",
    "sum_valor_indenizacao",
    "max_vl_subvencao_federal",
    "min_vl_subvencao_federal",
    "sum_vl_subvencao_federal",
    "max_pe_taxa",
    "min_pe_taxa",
    "sum_pe_taxa",
    "max_vl_premio_liquido",
    "min_vl_premio_liquido",
    "sum_vl_premio_liquido",
    "max_nr_produtividade_segurada",
    "min_nr_produtividade_segurada",
    "sum_nr_produtividade_segurada",
    "max_vl_limite_garantia",
    "min_vl_limite_garantia",
    "sum_vl_limite_garantia",
]
