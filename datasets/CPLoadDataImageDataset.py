from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import numpy as np
import pandas as pd
from pyarrow import parquet as pq

from .BaseImageDataset import BaseImageDataset
from ..transforms import (
    TransformType
    )

# alias for file index type checks that allow for
# str/pathlib paths to a .csv/.parquet file or 
# a pandas DataFrame directly
FileIndexType = Union[str, Path, pd.DataFrame]


class CPLoadDataImageDataset(BaseImageDataset):
    """
    Image dataset class that parses the `loaddata.csv` files, which are originally intended as
        inputs to the CellProfiler loaddata module, to initiate image analysis. To be precise,
        this dataset class is developed to handle the `loaddata.csv` files produced by pe2loaddata 
        (https://github.com/broadinstitute/pe2loaddata) from Phenix metadata XML files, so the 
        compatiblity with other loaddata formats is not guaranteed.
    Optionally accepts CellProfiler single cell level single cell level feature profile
        file(s) and extracts object level metadata from it.
    Due to its specialization, this dataset class may be removed from this repo in the future
        and re-located to a fork repository with a special dataset focus.
    
    Inheriting from the BaseImageDataset class, this class merely processeses the `loaddata.csv` 
        and the sc_profiles, builds file index, metadata and object level metadata accordingly, 
        then passes them to the parent class constructor. 
        From there, the parent class takes over performing the following functionalities:
        - Input/Target channel configuration
        - Image retrieval
        - Image transformation configuration
        - Ineternal state (last accessed image) tracking
        - Metadata handling
        - Dataset export/import
    """

    def __init__(
            self, 
            loaddata: FileIndexType,
            # TODO: support multiple sc_feature files/dfs
            sc_feature: Optional[FileIndexType], 
            # loaddata file parsing args, these should not change
            # so long as pe2loaddata parsing remains consistent
            loaddata_file_column_prefix: str = 'FileName_',
            loaddata_path_column_prefix: str = 'PathName_',
            # args below passed onto BaseImageDataset
            pil_image_mode: str = "I;16",
            input_channel_keys: Optional[Union[str, List[str]]] = None,
            target_channel_keys: Optional[Union[str, List[str]]] = None,
            input_transform: TransformType = None,
            target_transform: TransformType = None
        ):
            """
            Initialize the CPLoadDataImageDataset with the provided loaddata and sc_feature files.

            :param loaddata: Path to the loaddata CSV file or a pandas DataFrame containing the loaddata.
            :param sc_feature: Path to the CellProfiler single cell feature CSV file or a pandas DataFrame.
            :param loaddata_file_column_prefix: Prefix for the file name columns in the loaddata CSV.
            :param loaddata_path_column_prefix: Prefix for the path name columns in the loaddata CSV.
            :param pil_image_mode: PIL image mode for loading images.
            :param input_channel_keys: Keys for input channels, can be a single key or a list of keys.
            :param target_channel_keys: Keys for target channels, can be a single key or a list of keys.
            :param input_transform: Transform to apply to input images.
            :param target_transform: Transform to apply to target images.
            """
            
            self.__file_column_prefix = loaddata_file_column_prefix
            self.__path_column_prefix = loaddata_path_column_prefix
                        
            loaded_loaddata = self._load_loaddata_file(loaddata)
            # when sc_feature is supplied, this merges the sc_feature into loaddata
            # to produce the metadata dataframe, multi-entry matches from the merge 
            # are isolated as a list of DataFrames `object_metadata` and passed to the parent class
            metadata, object_metadata = self._load_merge_sc_feature_file(loaded_loaddata, sc_feature)
            # generate the file index DataFrame from the metadata
            file_index = self.__construct_file_index(metadata)

            super().__init__(
                file_index=file_index,
                metadata=metadata,
                # object_metadata=object_metadata,
                pil_image_mode=pil_image_mode,
                input_channel_keys=input_channel_keys,
                target_channel_keys=target_channel_keys,
                input_transform=input_transform,
                target_transform=target_transform
            )

            self.set_object_metadata(object_metadata=object_metadata)
    
    """
    Helper methods for parsing and validating the pe2loaddata loaddata csv files
    """
    def _validate_format_file_index(
            self,
            file_index: FileIndexType,
        ) -> pd.DataFrame:

        if file_index is None or isinstance(file_index, pd.DataFrame):
            return file_index
        
        if isinstance(file_index, str):
            # resolve the file path
            file_index = Path(file_index)
        elif isinstance(file_index, Path):
            pass
        else:
            raise TypeError(
                "Expected string, Path, or pandas DataFrame for file_index. "
                f"Received: {type(file_index)}"
            )
        
        if not file_index.exists() or not file_index.is_file():
            raise FileNotFoundError(f"File does not exist: {file_index}")
        elif file_index.suffix not in ('.csv', '.parquet'):
            raise ValueError(
                f"File must be a CSV or Parquet file, got: {file_index.suffix}"
            )
        
        return file_index
    
    def _load_loaddata_file(
            self, 
            loaddata: FileIndexType
        ) -> pd.DataFrame:
        """
        Load the loaddata file and validate it is non-empty.
        """
        # Catch and re-raise exceptions with more context from
        # helper method _validate_format_file_index that 
        # validates and formats the loaddata file index
        try:
            loaddata = self._validate_format_file_index(loaddata)
        except (FileNotFoundError, TypeError) as e:
            raise ValueError(
                f"Invalid loaddata file: {e}"
            ) from e
        except ValueError as e:
            raise ValueError(
                f"Error reading loaddata file: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error reading loaddata file: {e}"
            ) from e        
        if loaddata is None:
            raise ValueError("Required input loaddata file is None.")        
        
        # Load the loaddata file if not already provided as a DataFrame
        if isinstance(loaddata, pd.DataFrame):
            loaded = loaddata
        else:
            try:
                loaded = pd.read_csv(loaddata) if loaddata.suffix == '.csv' else pd.read_parquet(loaddata)
            except Exception as e:
                raise ValueError(
                    f"Error reading loaddata file: {e}"
                ) from e
        
        # Check if the loaddata DataFrame is empty
        if len(loaded) == 0:
            raise ValueError("Loaddata file is empty.")
        
        # de duplication just in case
        hashable_cols = [col for col in loaded.columns if pd.api.types.is_hashable(loaded[col].iloc[0])]
        if len(hashable_cols) == 0:
            raise ValueError("No hashable columns found in loaddata file.")
        loaded = loaded.drop_duplicates(subset=hashable_cols)
        loaded.reset_index(drop=True, inplace=True)
        
        return loaded
    
    """
    Helper methods for inferring channel keys from the loaddata DataFrame
    and constructing the file index DataFrame
    """

    def __infer_channel_keys(
            self,
            loaddata_df: pd.DataFrame
        ) -> set[str]:
        """
        """
        # Retrieve columns that indicate path and filename to image files
        file_columns = [col for col in loaddata_df if col.startswith(self.__file_column_prefix)]
        path_columns = [col for col in loaddata_df if col.startswith(self.__path_column_prefix)]

        if len(file_columns) == 0 or len(path_columns) == 0:
            raise ValueError('No path or file columns found in loaddata csv.')
        
        # Anything following the prefix should be the channel names
        file_channel_keys = [col.replace(self.__file_column_prefix, '') for col in file_columns]
        path_channel_keys = [col.replace(self.__path_column_prefix, '') for col in path_columns]
        channel_keys = set(file_channel_keys).intersection(set(path_channel_keys))

        if len(channel_keys) == 0:
            raise ValueError('No matching channel keys found between file and path columns.')
        
        return channel_keys
    
    def __construct_file_index(
            self,
            loaddata_df: pd.DataFrame,
        ) -> pd.DataFrame:

        channel_keys = self.__infer_channel_keys(loaddata_df)
        
        def safe_join(row):
            path = row[path_col]
            fname = row[file_col]
            if pd.isna(path) or pd.isna(fname):
                raise ValueError(
                    f"Missing path or filename in loaddata row: {row}"
                )
            return str(Path(str(path)) / str(fname))

        file_index_dict = {}
        for key in channel_keys:
            file_col = f'{self.__file_column_prefix}{key}'
            path_col = f'{self.__path_column_prefix}{key}'

            file_index_dict[key] = loaddata_df.apply(
                safe_join,
                axis=1
            )

        return pd.DataFrame(file_index_dict)
    
    """
    Helper method for processing the optional single cell feature file and
    merging it with the loaddata DataFrame to produce the metadata and object level metadata.
    """
        
    def _load_merge_sc_feature_file(
            self, 
            loaddata_df: pd.DataFrame,
            sc_feature: FileIndexType
        ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        """
        try:
            sc_feature = self._validate_format_file_index(sc_feature)
        except (FileNotFoundError, TypeError) as e:
            raise ValueError(
                f"Invalid sc_feature file: {e}"
            ) from e
        except ValueError as e:
            raise ValueError(
                f"Error reading sc_feature file: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error reading sc_feature file: {e}"
            ) from e
        
        if sc_feature is None:
            # if sc_feature is None, we just return the loaddata_df 
            # and an empty list of DataFrames with the same length as loaddata_df
            return loaddata_df, [pd.DataFrame()] * len(loaddata_df)
        elif isinstance(sc_feature, pd.DataFrame):
            sc_features_loaded = sc_feature
            merge_cols = list(set(loaddata_df.columns).intersection(set(sc_features_loaded.columns.to_list())))
        else:
            # first read only the column names from the file
            try:
                if sc_feature.suffix == '.csv':
                    sc_feature_cols = pd.read_csv(sc_feature, nrows=0).columns.to_list()
                elif sc_feature.suffix == '.parquet':
                    pq_file = pq.ParquetFile(sc_feature)
                    sc_feature_cols = pq_file.schema.names
            except Exception as e:
                raise ValueError(
                    f"Error reading sc_feature file: {e}"
                ) from e            
            # Instead of loading in the entire sc_feature file,
            # load only:
            # columns that overlap between the loaddata and sc_feature
            merge_cols = list(set(loaddata_df.columns).intersection(set(sc_feature_cols)))
            # columns that start with 'Metadata_'
            metadata_cols = [col for col in sc_feature_cols if col.startswith('Metadata_')]
            # TODO: if we switch to something loading .parquet faster like duckdb maybe we can avoid 
            # the hassle of needing to subset columns
            try:
                sc_features_loaded = pd.read_csv(sc_feature, usecols=merge_cols + metadata_cols) if sc_feature.suffix == '.csv' \
                    else pq.read_table(sc_feature, columns=merge_cols + metadata_cols).to_pandas()
            except Exception as e:
                raise ValueError(
                    f"Error reading in columns {merge_cols + metadata_cols} from sc_feature file: {e}"
                ) from e            

        # merge the loaddata and sc_feature DataFrames
        # isolate the multi-entry matches as a list of DataFrames
        try:
            (loaddata_df_merged, object_sc_features_list) = CPLoadDataImageDataset.split_and_merge_matches(
                df_left=loaddata_df,
                df_right=sc_features_loaded,
                merge_keys=list(merge_cols),
                drop_keys_in_matches=True
            )
        except Exception as e:
            # Catch any exceptions and re-raise with more context
            raise RuntimeError(
                f"Error merging loaddata and sc_feature DataFrames: {e}"
            ) from e
        
        return loaddata_df_merged, object_sc_features_list
    
    """
    Other helper method(s)
    """    
    @staticmethod
    def split_and_merge_matches(
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        merge_keys: list[str],
        drop_keys_in_matches: bool = True,
    ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        """
        For each row in `df_left`, find matching rows in `df_right` based on a list of merge_keys (columns),
            then, separately:
        - Merging scalar columns from df_right (1:1 or 1:0 matching per key) directly into df_left.
        - Extract multi-entry (1:n) matches from df_right as separate dataframes grouped by merge_keys,
            and organize as a list of DataFrames, with ordering matching that in the merged `df_left`.

        TODO: Maybe this is not the best location for this method, but we are in the middle of a refactor so
        this will be here for now.

        :param df_left: DataFrame to merge into, containing the keys to match on.
        :param df_right: DataFrame to match against, containing the keys and scalar columns.
        :param merge_keys: List of column names to use as keys for matching.
        :param drop_keys_in_matches: Whether to drop the merge keys from the matched DataFrames.
        """
        if not isinstance(df_left, pd.DataFrame):
            raise TypeError(f"Expected df_left to be a pandas DataFrame. Got {type(df_left)}.")
        if not isinstance(df_right, pd.DataFrame):
            raise TypeError(f"Expected df_right to be a pandas DataFrame. Got {type(df_right)}.")
        
        if not isinstance(merge_keys, list) or not all(isinstance(key, str) for key in merge_keys):
            raise TypeError(f"Expected merge_keys to be a list of strings. Got {type(merge_keys)}.")

        if not all(key in df_left.columns for key in merge_keys):
            raise ValueError(f"Not all merge keys {merge_keys} found in df_left columns.")
        if not all(key in df_right.columns for key in merge_keys):
            raise ValueError(f"Not all merge keys {merge_keys} found in df_right columns.")
        
        if len(df_left) == 0:
            raise ValueError("df_left is empty. Cannot perform merge.")
        if len(df_right) == 0:
            raise ValueError("df_right is empty. Cannot perform merge.")

        # Index df_right for fast lookups
        df_right_indexed = df_right.set_index(merge_keys).sort_index()

        # Identify scalar columns (same value across duplicates per key)
        grouped = df_right.groupby(merge_keys)
        scalar_cols = [
            col for col in df_right.columns if col not in merge_keys
            and all(grouped[col].nunique(dropna=False) <= 1)
        ]

        # Pre-construct scalar data for left merge
        df_right_scalar = df_right[merge_keys + scalar_cols].drop_duplicates(subset=merge_keys)

        # Initialize output list
        match_list = []

        for _, row in df_left.iterrows():
            key = tuple(row[merge_keys])
            if key in df_right_indexed.index:
                matched = df_right_indexed.loc[key]
                if isinstance(matched, pd.Series):  # only one match
                    matched = matched.to_frame().T

                if drop_keys_in_matches:
                    matched = matched.drop(columns=merge_keys, errors="ignore")

                match_list.append(matched.reset_index(drop=True))
            else:
                match_list.append(pd.DataFrame())

        # Merge scalar values into df_left
        df_merged = df_left.merge(df_right_scalar, on=merge_keys, how='left')

        return df_merged, match_list