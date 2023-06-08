from config import DATA_PATH
import pandas as pd
from enum import Enum

from typing import List


class FileType(Enum):
    AH = 'ah'
    Y = 'y'


class GeographicScale(Enum):
    COUNTRY = 'countries_selected'
    REGION = 'regions'


class Data:
    """
    Base class for data

    """
    def __init__(self, year: int, geo_scale: GeographicScale, file_type: FileType):
        self.path = f'{DATA_PATH}/technology/{geo_scale.value}/{year}-mat_{file_type.value}.csv'
        self.year = year
        self.file_type = file_type
        self.data = pd.read_csv(self.path, index_col=0)


class DataMean:
    def __init__(self, years: List[int], geo_scale: GeographicScale, file_type: FileType):
        self.year = years
        self.paths = [f'{DATA_PATH}/technology/{geo_scale.value}/{year}-mat_{file_type.value}.csv' for year in self.year]
        self.file_type = file_type
        self.datasets = [pd.read_csv(path, index_col=0) for path in self.paths]
        self.data = 0
        for dataset in self.datasets:
            self.data += dataset

        self.data /= len(self.datasets)


if __name__ == '__main__':
    d = DataMean(years=[2017, 2018, 2019], geo_scale=GeographicScale.COUNTRY, file_type=FileType.Y)
    d