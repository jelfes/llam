import sqlite3
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from config_local import DATA_DIR


class NewsDatasetSQL(Dataset):
    """
    This dataset contains news articles and outputs them as part of a prompt
    for an LLM. The prompt instructs the LLM to extract actants according to
    Greimas' Actantial Model from the articles.
    """

    def __init__(
        self,
        query,
    ):
        conn = sqlite3.connect(Path(DATA_DIR, "database.db"))
        self.table = pd.read_sql(query, conn)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """
        Retrives an item from the dataset.

        Parameters:
            idx: index to select the item.
        """

        article = self.table.loc[idx, "text"]

        return article
