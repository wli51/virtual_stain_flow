from pathlib import Path
from typing import Sequence, Optional, Tuple, Union
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from .base_dataset import BaseImageDataset

"""
Dataset class creation from CellProfiler loaddata CSV and optionally
single-cell features Parquet files.

This class simply wraps BaseImageDataset by calling helper methods to
wrangle the loaddata and sc_feature metadata to yield the file_index,
metadata and object_metadata parameters required/received by the 
BaseImageDataset.
"""
class CPLoadDataImageDataset(BaseImageDataset):    
    
    def __init__(
        self, 
        loaddata: Union[Path, pd.DataFrame],
        sc_feature: Optional[Sequence[Path]]=None,
        **kwargs
    ):
        """
        Initializes the dataset

        :param loaddata: Path to the loaddata CSV file or a DataFrame.
        :param sc_feature: Optional sequence of paths to single-cell features 
            Parquet files or a DataFrame containing single-cell features.
        :param kwargs: Additional keyword arguments passed to BaseImageDataset.
        """

        # helper method loads from file if not already a DataFrame
        loaddata = load_loaddata(loaddata)

        # helper method loads from a list of Parquet files if not 
        # already a DataFrame, or returns None if sc_feature is None.
        # the helper method also handles the column selection to ensure
        # multiple Parquet files can be loaded and combined
        # into a single DataFrame.
        sc_feature = load_sc_features(sc_feature)
        
        if sc_feature is not None:
            # merge the loaddata with the sc_feature DataFrame
            # this yields a DataFrame with the unique scalar features
            # per view/site, and a list of DataFrames with the
            # object level metadata that corresponds to the non-unique
            # per view/site metadata from sc_feature, which will serve
            # as the metadata and object_metadata input to BaseImageDataset
            # initialization, respectively.
            loaddata, sc_feature_by_view = merge_loaddata_sc_feature(
                loaddata,
                sc_feature
            )
        else:
            # if no sc_feature is provided, create dummy object_metadata
            # the loaddata still serves as the metadata
            sc_feature_by_view = [pd.DataFrame() for _ in range(len(loaddata))]

        # format the loaddata DataFrame to yield the file_index
        file_index = get_channel_file_index(
            loaddata,
            path_prefix='PathName_',
            file_prefix='FileName_'
        )

        # Pass wrangled dataframes to BaseImageDataset
        # the base class takes over complete from here, hence this class 
        # implements no further dataset logic. 
        super().__init__(
            file_index=file_index,
            metadata=loaddata,
            object_metadata=sc_feature_by_view,
            **kwargs
        )       

"""
Helper methods for wrangling the loaddata and sc_feature dataframes
"""

def load_loaddata(
    loaddata: Union[Path, pd.DataFrame]
):
    """
    Load the loaddata from a file or DataFrame.
    Handles parameter and file validation and returns the loaded DataFrame. 

    :param loaddata: Path to the loaddata CSV file or a DataFrame.
    """
    if loaddata is None:
        raise ValueError("loaddata cannot be None.")

    if isinstance(loaddata, pd.DataFrame):
        if loaddata.empty:
            raise ValueError("loaddata DataFrame cannot be empty.")
        return loaddata

    if isinstance(loaddata, str):
        loaddata = Path(loaddata)
    if isinstance(loaddata, Path):
        if not loaddata.exists():
            raise FileNotFoundError(f"File not found: {loaddata}")
        if not loaddata.is_file():
            raise ValueError(f"Expected a file, got: {loaddata}")
        try:
            if loaddata.suffix == '.csv':
                loaddata_df = pd.read_csv(loaddata)
            elif loaddata.suffix == '.parquet':
                loaddata_df = pd.read_parquet(loaddata)
            else:
                raise ValueError(
                    f"Expected a CSV or Parquet file, got: {loaddata.suffix}"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load loaddata from {loaddata}: {e}"
            ) from e
        
        return loaddata_df

def load_sc_features(
    sc_features: Union[Sequence[Path], pd.DataFrame]
) -> pd.DataFrame:
    """
    Load single-cell features from a parquet file or DataFrame.
    Handles parameter and file validation, 
        actual loading is done by `load_combine_sc_features`.
    """
    
    if sc_features is None:
        return None
    
    if isinstance(sc_features, pd.DataFrame):
        if sc_features.empty:
            raise ValueError("sc_features DataFrame cannot be empty.")
        return sc_features
    
    if isinstance(sc_features, str):
        sc_features = [Path(sc_features)]
    if isinstance(sc_features, Path):
        sc_features = [sc_features]
    if isinstance(sc_features, Sequence):
        if not all(isinstance(f, Path) for f in sc_features):
            raise TypeError(
                "All elements in sc_features must be Path objects."
            )
        if not sc_features:
            raise ValueError("sc_features cannot be an empty sequence.")
        
        return load_combine_sc_features(sc_features)
    
    else:
        raise TypeError(
            "Expected sc_features to be a DataFrame or a sequence of Path objects, "
            f"got {type(sc_features).__name__} instead."
        )
    
def load_combine_sc_features(
    files: Sequence[Path],
    col_filter_prefix='Metadata_'
):
    """
    Load and combine single-cell features from single/multiple Parquet files.
    
    :param files: List of file paths to Parquet files.
    :return: Combined DataFrame of single-cell features.
    """
    parquet_handles = []
    parquet_columns = []

    from pyarrow import parquet as pq

    for file in files:

        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
        if not file.is_file():
            raise ValueError(f"Expected a file, got: {file}")
        if not file.suffix == '.parquet':
            raise ValueError(f"Expected a Parquet file, got: {file.suffix}")        

        handle = pq.ParquetFile(file)
        parquet_handles.append(handle)
        parquet_columns.append(handle.schema.names)

    shared_columns = list(set.intersection(*map(set, parquet_columns)))
    shared_columns = [
        col for col in shared_columns if col.startswith(col_filter_prefix)
    ]
    if not shared_columns:
        raise ValueError("No shared columns found with the specified prefix.")
    # preserve the order
    shared_columns = [col for col in parquet_columns[0] if col in shared_columns]
    combined_df = pd.concat([
        handle.read(columns=shared_columns).to_pandas() for handle in parquet_handles
    ], ignore_index=True, axis=0)

    return combined_df

def get_channel_file_index(
    loaddata_df: pd.DataFrame,
    path_prefix: str = 'PathName_',
    file_prefix: str = 'FileName_'
)-> pd.DataFrame:
    """
    Get a DataFrame mapping channel keys to file paths and names.
    This function infers channel keys from the DataFrame columns and constructs
    a new DataFrame with the file paths and names for each channel.
    :param loaddata_df: DataFrame containing file paths and names.
    :param path_prefix: Prefix for path columns.
    :param file_prefix: Prefix for file name columns.
    :return: DataFrame with channel keys as columns and file paths as values.
        Formatted to serve as the file_index input to `BaseImageDataset` class.
    """
    
    channel_keys = infer_channel_keys(
        loaddata_df, 
        path_prefix=path_prefix, 
        file_prefix=file_prefix
    )
    if len(channel_keys) == 0:
        raise ValueError("No channel keys found in the DataFrame.")

    file_index_df = pd.DataFrame(
        columns=channel_keys, 
        index=loaddata_df.index
    )
    for channel in channel_keys:
        file_index_df[channel] = loaddata_df.apply(
            lambda row: Path(row[f'{path_prefix}{channel}']) /\
                str(row[f'{file_prefix}{channel}']), axis=1)
        
    return file_index_df

def infer_channel_keys(
    loaddata_df: pd.DataFrame,
    path_prefix: str = 'PathName_',
    file_prefix: str = 'FileName_'
) -> List[str]:
    """
    Infer channel keys from the DataFrame columns.
    
    :param loaddata_df: DataFrame containing file paths and names.
    :param path_prefix: Prefix for path columns.
    :param file_prefix: Prefix for file name columns.
    :return: List of inferred channel keys.
    """
    filename_cols = [col for col in loaddata_df.columns if col.startswith(file_prefix)]
    filepath_cols = [col for col in loaddata_df.columns if col.startswith(path_prefix)]
    
    channel_strip_name_cols = [
        col.replace(file_prefix, '') for col in filename_cols
    ]    
    channel_strip_path_cols = [
        col.replace(path_prefix, '') for col in filepath_cols
    ]
    
    return list(set(channel_strip_name_cols) & set(channel_strip_path_cols))

def merge_loaddata_sc_feature(
    loaddata_df: pd.DataFrame,
    sc_features_df: pd.DataFrame,
    merge_fields: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Merge loaddata DataFrame with single-cell features from cell profiler,
    yielding two outputs:
        1. A 1 on 1 merge of the loaddata_df with the columns from sc_features_df
            that are unique per view/site,
        2. A list of DataFrames, each containing the object level metadata which
            corresponds to the non-unique per view/site metadata from sc_features_df.
            The list is ordered such that the i-th element corresponds
            to the i-th row of the merged DataFrame.

    :param loaddata_df: DataFrame containing loaddata.
    :param sc_features_df: DataFrame containing single-cell features.
    :param merge_fields: List of fields to merge on. If None, will infer from
        the intersection of columns in both DataFrames.
    :return: Tuple containing:
        - Merged DataFrame with unique scalar features.
        - List of DataFrames containing object level metadata.
    """
    
    # Infer merge fields if not provided
    if merge_fields is None:
        merge_fields = list(
            set(loaddata_df.columns).intersection(set(sc_features_df.columns)))
    elif isinstance(merge_fields, Sequence):
        if not all(isinstance(field, str) for field in merge_fields):
            raise TypeError(
                "All elements in merge_fields must be strings."
            )
        if not all(field in loaddata_df.columns for field in merge_fields):
            raise ValueError(
                "Some fields in merge_fields are not present in loaddata_df."
            )
    else:
        raise TypeError(
            "Expected merge_fields to be a list or None, "
            f"got {type(merge_fields).__name__} instead."
        )        
    
    if not merge_fields:
        raise ValueError("No common fields found to merge on.")    

    # sort for faster look ups
    # here we assume the merge fields also serve as a unique site identifier
    loaddata_indexed = loaddata_df.set_index(merge_fields).sort_index()
    sc_feature_indexed = sc_features_df.set_index(merge_fields).sort_index()
    sc_feature_grouped = sc_feature_indexed.groupby(merge_fields)

    # isolate the columns from sc_features that is unique to each view/site
    # (only 1 unqiue value when grouped by merge_fields)
    # excluding the merge fields
    sc_feature_scalar_cols = [
        col for col in sc_features_df.columns if col not in merge_fields
        and all(sc_feature_grouped[col].nunique(dropna=False) <= 1)
    ]    
    sc_feature_scalar = sc_feature_indexed[
        sc_feature_scalar_cols].drop_duplicates()

    # merge the unique per view/site scalar columns from sc_features_df
    # with loaddata_df, dropping all view/site rows that do not match
    # between loaddata and sc_features
    scalar_merged_df = loaddata_indexed.merge(
        sc_feature_scalar, 
        how='inner',
        left_index=True,
        right_index=True
    )

    # collect the non-unique per view/site metadata entries that
    # should represent object level metadata from cellprofiler analysis
    # as a list of DataFrames, while ensuring the length and order
    # of the list matches scalar_merged_df rows. 
    sc_feature_object_list = []
    for key, row in scalar_merged_df.iterrows():
        key = tuple(key)
        if key in sc_feature_indexed.index:
            matched = sc_feature_indexed.loc[key]
            if isinstance(matched, pd.Series):
                matched = matched.to_frame().T
            matched = matched.drop(
                columns=merge_fields,
                errors='ignore'
            )
            matched.reset_index(drop=False, inplace=True)
            sc_feature_object_list.append(matched)
        else:
            # append placeholder if no object level 
            # metadata is associated with a view, this ensures
            # sc_feature_object_list is the same length as loaddata_df
            sc_feature_object_list.append(pd.DataFrame())
            
    scalar_merged_df.reset_index(inplace=True, drop=False)

    return scalar_merged_df, sc_feature_object_list