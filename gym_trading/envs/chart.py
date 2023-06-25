"""
This module provides classes to define charts with dates and values.
"""

from typing import Optional, List

import pandas as pd


class ChartDataFrame(pd.DataFrame):
    """
    A custom DataFrame class for working with chart data.

    This class extends the pandas DataFrame class and provides additional methods
    for handling chart data, such as retrieving values at a specific date,
    calculating profit, and filtering data based on a date window.

    Attributes:
        _metadata (list): A list of metadata attributes.

    Methods:
        __init__(*args, **kwargs): Initializes the ChartDataFrame.
        value_at(date) -> Optional[float]: Retrieves the value at a specific date.
        add_data(date, value): Adds a new row with the specified date and value.
        with_data_window_until_date(end_date, window) -> ChartDataFrame:
            Filters the data based on a date window.
        profit() -> float: Calculates the profit based on the first and last values.
        dates() -> List: Returns a list of dates in the DataFrame.
        values() -> List: Returns a list of values in the DataFrame.
    """
    _metadata = ['DATE', 'VALUE']

    def __init__(self, *args, **kwargs):
        """
        Initializes the ChartDataFrame.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._validate()

    def _validate(self):
        """
        Validates the ChartDataFrame.

        Raises:
            ValueError: If 'DATE' or 'VALUE' columns are missing.
        """
        # Check if 'date' column is present
        if 'DATE' not in self.columns:
            raise ValueError("Column 'DATE' is required.")

        # Check if 'value' column is present
        if 'VALUE' not in self.columns:
            raise ValueError("Column 'VALUE' is required.")

    def value_at(self, date) -> Optional[float]:
        """
        Retrieves the value at a specific date.

        Args:
            date: The date to retrieve the value for.

        Returns:
            Optional[float]: The value at the specified date, or None if not found.
        """
        value = self.loc[self['DATE'] == date, 'VALUE']
        return value.values[0] if not value.empty else None

    def add_data(self, date, value):
        """
        Adds a new row with the specified date and value.

        Args:
            date: The date to add.
            value: The value to add.

        Raises:
            ValueError: If the new row is missing the 'DATE' or 'VALUE' columns.
        """
        new_row = pd.DataFrame({'DATE': [date], 'VALUE': [value]})
        self._validate_new_row(new_row)
        self._add_new_row(new_row)

    @staticmethod
    def _validate_new_row(new_row):
        """
        Validates a new row.

        Args:
            new_row: The new row to validate.

        Raises:
            ValueError: If the new row is missing the 'DATE' or 'VALUE' columns.
        """
        # Check if the new row has the required columns
        if not {'DATE', 'VALUE'}.issubset(new_row.columns):
            raise ValueError(
                "New row must contain columns 'DATE' and 'VALUE'.")

    def _add_new_row(self, new_row):
        """
        Adds a new row to the ChartDataFrame.

        Args:
            new_row: The new row to add.
        """
        self.loc[len(self)] = new_row.iloc[0]

    def with_data_window_until_date(self, end_date, window):
        """
        Filters the data based on a date window.

        Args:
            end_date: The end date for the data window.
            window: The number of rows to include in the data window.

        Returns:
            ChartDataFrame: A new ChartDataFrame instance with the filtered data.
        """
        # Sort the DataFrame by date in descending order
        sorted_data = self.sort_values(by='DATE', ascending=False)

        # Convert sorted_data to a DataFrame if it's not already
        if not isinstance(sorted_data, pd.DataFrame):
            sorted_data = pd.DataFrame(sorted_data)

        # Filter the rows based on the date condition and select the last n
        # values
        filtered_data = sorted_data[sorted_data['DATE']
                                    <= end_date].head(window)

        # Sort the filtered data by date in ascending order
        filtered_data = filtered_data.sort_values(by='DATE')

        # Create a new ChartDataFrame instance with the filtered data
        new_chart_data = ChartDataFrame(filtered_data)

        return new_chart_data

    def profit(self):
        """
        Calculates the profit based on the first and last values.

        Returns:
            float: The calculated profit.
        """
        try:
            values = self['VALUE'].tolist()
            first_value = values[0]
            last_value = values[len(values) - 1]
            return ((last_value / first_value) - 1) * 100
        except IndexError:
            return 0

    @property
    def dates(self) -> List:
        """
        Returns a list of dates in the DataFrame.

        Returns:
            List: A list of dates.
        """
        return self['DATE'].tolist()

    @property
    def values(self) -> List:
        """
        Returns a list of values in the DataFrame.

        Returns:
            List: A list of values.
        """
        return self['VALUE'].tolist()
