from mosden.base import BaseClass
import pandas as pd
import os


class Reprocessing(BaseClass):
    def __init__(self, input_path: str) -> None:
        """
        This class holds data from the literature for chemical removal rates.

        Parameters
        ----------
        input_path : str
            Path to the input file containing relevant data paths
        """
        super().__init__(input_path)
        return None

    def removal_scheme(self, rate_csv: str='MSBR.csv', include_long: bool=True,
                    rate_scaling: float=1.0) -> dict[str, float]:
        """
        Returns the removal scheme with rate scaling and optional longer cycle time
        elemental removal included

        Parameters
        ----------
        rate_csv : str
            The name of the removal rate CSV file to use (defaults to MSBR.csv)
        include_long : bool
            True to include long cycle time elements (defaults to True)
        rate_scaling : float
            The scaling to apply to removal rates (defaults to 1.0)
        
        Returns
        -------
        dict[str, float]
            Elemental chemical removal rates [per second]
        """
        repr_dict = {}
        repr_dir: str = os.path.join(self.repr_dir, rate_csv)
        repr_data: pd.DataFrame = pd.read_csv(repr_dir)
        repr_dict_dirty: dict = repr_data.to_dict(index=False, orient='tight')
        repr_dict_clean: list[list] = repr_dict_dirty['data']
        for element, rate in repr_dict_clean:
            if (not include_long) and (rate < 0.05):
                continue
            repr_dict[element] = float(rate) * rate_scaling
        
        return repr_dict
    
if __name__ == '__main__':
    input_path = "../../examples/huynh_2014/input.json"
    repr = Reprocessing(input_path)
    data = repr.removal_scheme('MSBR.csv')
    print(data)
    