"""
Data loader for accessing market data from balerion-data repository.

This module provides utilities to load OHLCV data from the parquet files
stored in the balerion-data repository.
"""

from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np


class DataLoader:
    """Load market data from balerion-data parquet files."""

    def __init__(self, data_root: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.

        Args:
            data_root: Path to balerion-data/data directory.
                      Defaults to ../balerion-data/data relative to this repo.
        """
        if data_root is None:
            # Default to sibling directory
            current_dir = Path(__file__).resolve().parent.parent
            data_root = current_dir.parent / "balerion-data" / "data"

        self.data_root = Path(data_root)

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_root}\n"
                "Please ensure balerion-data repository is set up correctly."
            )

        self.fx_dir = self.data_root / "fx"
        self.indices_dir = self.data_root / "indices"

    def load_fx(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """
        Load FX pair data.

        Args:
            symbol: FX pair symbol (e.g., 'EURUSD', 'USDJPY')
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
            timeframe: Timeframe of data (default: '1m')

        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        filepath = self.fx_dir / f"{symbol.lower()}_{timeframe}.parquet"
        return self._load_and_filter(filepath, start_date, end_date)

    def load_index(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """
        Load index data.

        Args:
            symbol: Index symbol (e.g., 'US30', 'XAUUSD')
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
            timeframe: Timeframe of data (default: '1m')

        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        filepath = self.indices_dir / f"{symbol.lower()}_{timeframe}.parquet"
        return self._load_and_filter(filepath, start_date, end_date)

    def load_multiple(
        self,
        symbols: List[str],
        asset_type: str = "fx",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1m",
    ) -> dict[str, pd.DataFrame]:
        """
        Load multiple symbols at once.

        Args:
            symbols: List of symbol names
            asset_type: Either 'fx' or 'indices'
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
            timeframe: Timeframe of data (default: '1m')

        Returns:
            Dictionary mapping symbol names to DataFrames
        """
        load_func = self.load_fx if asset_type == "fx" else self.load_index

        data = {}
        for symbol in symbols:
            try:
                data[symbol] = load_func(symbol, start_date, end_date, timeframe)
            except FileNotFoundError:
                print(f"Warning: Data not found for {symbol}, skipping...")

        return data

    def _load_and_filter(
        self,
        filepath: Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load parquet file and optionally filter by date range.

        Args:
            filepath: Path to parquet file
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)

        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_parquet(filepath)

        # Ensure timestamp is datetime and set as index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        # Filter by date range if specified
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]

        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    def get_available_fx_symbols(self) -> List[str]:
        """Get list of available FX symbols."""
        return self._get_available_symbols(self.fx_dir)

    def get_available_index_symbols(self) -> List[str]:
        """Get list of available index symbols."""
        return self._get_available_symbols(self.indices_dir)

    def _get_available_symbols(self, directory: Path) -> List[str]:
        """Get list of symbols from parquet files in directory."""
        if not directory.exists():
            return []

        symbols = []
        for file in directory.glob("*.parquet"):
            # Extract symbol from filename (e.g., 'eurusd_1m.parquet' -> 'EURUSD')
            symbol = file.stem.split("_")[0].upper()
            symbols.append(symbol)

        return sorted(symbols)

    def resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe.

        Args:
            df: DataFrame with OHLCV data (must have timestamp index)
            timeframe: Target timeframe (e.g., '5T', '15T', '1H', '4H', '1D')
                      T = minutes, H = hours, D = days

        Returns:
            Resampled DataFrame
        """
        resampled = df.resample(timeframe).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Handle optional columns if they exist
        if "spread" in df.columns:
            resampled["spread"] = df["spread"].resample(timeframe).mean()
        if "real_volume" in df.columns:
            resampled["real_volume"] = df["real_volume"].resample(timeframe).sum()

        # Remove rows with NaN (incomplete periods)
        resampled.dropna(inplace=True)

        return resampled

    def get_data_info(self, symbol: str, asset_type: str = "fx") -> dict:
        """
        Get information about available data for a symbol.

        Args:
            symbol: Symbol name
            asset_type: Either 'fx' or 'indices'

        Returns:
            Dictionary with data information (rows, date range, file size, etc.)
        """
        load_func = self.load_fx if asset_type == "fx" else self.load_index

        try:
            df = load_func(symbol)

            directory = self.fx_dir if asset_type == "fx" else self.indices_dir
            filepath = directory / f"{symbol.lower()}_1m.parquet"
            file_size_mb = filepath.stat().st_size / (1024 * 1024)

            return {
                "symbol": symbol,
                "asset_type": asset_type,
                "rows": len(df),
                "start_date": df.index.min(),
                "end_date": df.index.max(),
                "file_size_mb": round(file_size_mb, 2),
                "columns": df.columns.tolist(),
            }
        except FileNotFoundError:
            return {"symbol": symbol, "error": "Data not found"}


# Convenience function for quick loading
def load_data(
    symbol: str,
    asset_type: str = "fx",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_root: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Quick function to load data for a single symbol.

    Args:
        symbol: Symbol name (e.g., 'EURUSD', 'US30')
        asset_type: Either 'fx' or 'indices'
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        data_root: Path to data directory (optional)

    Returns:
        DataFrame with OHLCV data

    Example:
        >>> df = load_data('EURUSD', start_date='2024-01-01', end_date='2024-12-31')
        >>> print(df.head())
    """
    loader = DataLoader(data_root)

    if asset_type == "fx":
        return loader.load_fx(symbol, start_date, end_date)
    else:
        return loader.load_index(symbol, start_date, end_date)
