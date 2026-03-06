import pandas as pd
import os
from uncertainties import ufloat
import numpy as np
import logging
from collections import defaultdict

class CSVHandler:
    def __init__(self, file_path: str, overwrite: bool=False, create=True) -> None:
        """
        This class reads and writes CSV files for MoSDeN data.

        Parameters
        ----------
        file_path : str
            Path to the CSV file
        overwrite : bool, optional
            Whether to overwrite the file if it exists, by default False
        create : bool, optional
            Whether to create the file if it does not exist, by default True
        """
        self.file_path = file_path
        if create:
            self._create_directory() 
        self.overwrite = overwrite
        self.logger = logging.getLogger(__name__)
        return None
    
    def _create_directory(self) -> None:
        """
        Create the directory for the file path if it does not exist.
        """
        directory = os.path.dirname(self.file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return None
    
    def _file_exists(self) -> bool:
        """
        Check if the file exists at the specified path.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        return os.path.isfile(self.file_path)

    def read_csv(self, raw_iaea=False) -> dict[str, dict[str, float]]:
        """
        Read the CSV file and return the data as a dictionary.

        Parameters
        ----------
        raw_iaea : bool, optional
            Whether the data is in IAEA format, by default False

        Returns
        -------
        dict[str, dict[str, float]]
            The data read from the CSV file
        """
        if raw_iaea:
            return self._read_iaea_csv()
        df = pd.read_csv(self.file_path, index_col=0)
        data = df.to_dict(orient='index')
        return data

    def read_csv_with_time(self, trim:bool = False) -> dict[str, dict[float, tuple[float, float]]]:
        """
        Read the CSV file and return the data as a dictionary.

        Parameters
        ----------
        trim : bool 
            Indicates if the time column should be removed (only if a single 
            time value is used)

        Returns
        -------
        dict[str, dict[float, (float, float)]]
            The data read from the CSV file
        """
        df = pd.read_csv(self.file_path)
        data = defaultdict(dict)
        is_single_time = df['Time'].nunique() == 1
        for _,row in df.iterrows():
            if is_single_time and trim:
                data[row.Nuclide] = (row.Concentration, row['sigma Concentration'])
            else:
                data[row.Nuclide][row.Time] = (row.Concentration, row['sigma Concentration'])
        return data
    
    def _read_iaea_csv(self) -> dict[str, dict[str, float]]:
        """
        Read the IAEA CSV file and return the data as a dictionary.

        Returns
        -------
        dict
            The data read from the IAEA CSV file.
        """
        data: dict[str, dict[str, float]] = {}
        df = pd.read_csv(self.file_path, header=1)
        for _, row in df.iterrows():
            iaea_nuc = row['nucid']
            nuc = self._iaea_to_mosden_nuc(iaea_nuc)
            half_life = row[' T1/2 [s] ']
            half_life_uncertainty = row[' D T1/2 [s]']
            if row['D pn1'] < 0:
                row['D pn1'] = 0
            if row['D pn2 '] < 0:
                row['D pn2 '] = 0
            if row['D pn3'] < 0:
                row['D pn3'] = 0
            mult_factor:int = 100
            P1 = ufloat(row[' pn1 % ']/mult_factor, row['D pn1']/mult_factor)
            P2 = ufloat(row[' pn2 % ']/mult_factor, row['D pn2 ']/mult_factor)
            P3 = ufloat(row[' pn3 % ']/mult_factor, row['D pn3']/mult_factor)
            prob_beta = ufloat(row['  beta- %']/mult_factor, row[' D beta-']/mult_factor)
            emission_prob = (1*P1 + 2*P2 + 3*P3) * prob_beta
            if np.isclose(emission_prob, 0.0):
                continue
            data[nuc] = {}
            data[nuc]['half_life'] = half_life
            data[nuc]['sigma half_life'] = half_life_uncertainty
            data[nuc]['emission probability'] = emission_prob.n
            data[nuc]['sigma emission probability'] = emission_prob.s
        return data
    
    def _iaea_to_mosden_nuc(self, iaea_nuc: str) -> str:
        """
        Converts IAEA nuclide format to MoSDeN format.

        Parameters
        ----------
        iaea_nuc : str
            IAEA nuclide identifier

        Returns
        -------
        str
            MoSDeN formatted nuclide identifier (e.g., 'U235', 'Br87')
        """
        i = 0
        while i < len(iaea_nuc) and iaea_nuc[i].isdigit():
            i += 1
        mass = iaea_nuc[:i]
        element = iaea_nuc[i:].capitalize()
        return f"{element}{mass}"

    def write_csv(self, data: dict[str: dict[str, float]]) -> None:
        """
        Write the data to a CSV file.

        Parameters
        ----------
        data : dict[str: dict[str, float]]
            The data to write to the CSV file, where keys are nuclides and values are dictionaries of properties.
        """
        if not self.overwrite and self._file_exists():
            self.logger.warning(f"File {self.file_path} already exists. Set overwrite=True to overwrite.")
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = 'Nuclide'
        df.to_csv(self.file_path, index=True)
        return None
    
    def write_csv_with_time(self, data: dict[str, dict[str, float]]) -> None:
        """
        Write the time-dependent data to a CSV file.

        Parameters
        ----------
        data : dict[str, dict[str, float]]
            The data to write to the CSV file, listed over time
        """
        if not self.overwrite and self._file_exists():
            self.logger.warning(f"File {self.file_path} already exists. Set overwrite=True to overwrite.")
        df = pd.DataFrame(data)
        df.to_csv(self.file_path, index=False)
        return None
    
    def write_groups_csv(self, data: dict[str: list[float]], sortby: str = 'half_life') -> None:
        """
        Write the group data to a CSV file, sorted by a specified column.

        Parameters
        ----------
        data : dict[str: list[float]]
            The data to write, where keys are group names and values are lists of properties.
        sortby : str, optional
            The column to sort, by default 'half_life'
        """
        if not self.overwrite and self._file_exists():
            self.logger.warning(f"File {self.file_path} already exists. Set overwrite=True to overwrite.")
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.sort_values(by=sortby, ascending=False)
        df.to_csv(self.file_path, index=False)
        return None

    
    def write_count_rate_csv(self, data: dict[str: list[float]]) -> None:
        """
        Write the count rate data to a CSV file.

        Parameters
        ----------
        data : dict[str: list[float]]
            The data to write, where keys are nuclides and values are lists of count rates
        """
        if not self.overwrite and self._file_exists():
            self.logger.warning(f"File {self.file_path} already exists. Set overwrite=True to overwrite.")
        df = pd.DataFrame(data)
        df.to_csv(self.file_path, index=False)
        return None
    
    def read_vector_csv(self) -> dict[str: list[float]]:
        """
        Read a vector CSV file and return the data as a dictionary.
        The data is structured such that each column is a key and the values are lists.

        Returns
        -------
        dict[str: list[float]]
            The vector data from the CSV file
        """
        df = pd.read_csv(self.file_path)
        data = {col: df[col].astype(float).tolist() for col in df.columns}
        return data