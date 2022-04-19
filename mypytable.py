"""
Programmer: Rie Durnil
Class: CPSC 322-01, Spring 2022
Programming Assignment #6
3/31/22

Description: This program defines the MyPyTable class which is used to store, clean, and
manipulate data for data mining.
"""

import copy
import csv
from tabulate import tabulate # uncomment if you want to use the pretty_print() method

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = []
        index = self.column_names.index(col_identifier)
        for row in self.data:
            if not include_missing_values and row[index] == "NA":
                pass
            else:
                col.append(row[index])
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort(reverse=True)
        for row in row_indexes_to_drop:
            self.data.pop(row)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        infile = open(filename, "r")
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            self.data.append(row)
        infile.close()
        self.column_names = self.data.pop(0)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(self.column_names)
        csv_writer.writerows(self.data)
        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        cols = []
        keys = []
        duplicate_indexes = []
        non_duplicates = []
        for col_name in key_column_names:
            cols.append(self.get_column(col_name, True))

        # all columns should be same length since we included missing values
        for i in range(len(cols[0])):
            keys.append([])
            for j in range(len(cols)):
                keys[i].append(cols[j][i])

        for index in range(len(keys)):
            if keys[index] in non_duplicates:
                duplicate_indexes.append(index)
            else:
                non_duplicates.append(keys[index])

        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        to_remove = []
        for i in range(len(self.data)):
            for j in range(len(self.column_names)):
                if self.data[i][j] == "NA":
                    if len(to_remove) == 0 or to_remove[-1] != i:
                        to_remove.append(i)
        to_remove.sort(reverse=True)
        for index in to_remove:
            self.data.pop(index)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col = self.get_column(col_name, False)
        index = self.column_names.index(col_name)
        average = sum(col) / len(col)
        for row in self.data:
            if row[index] == "NA":
                row[index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        summary_table = []
        for i in range(len(col_names)):
            col = self.get_column(col_names[i])
            if len(col) > 0:
                remove_missing_values(col)
                col.sort()
                summary_table.append([])
                summary_table[i].append(col_names[i])
                summary_table[i].append(col[0])
                summary_table[i].append(col[-1])
                summary_table[i].append((col[-1] + col[0]) / 2)
                summary_table[i].append(sum(col) / len(col))
                if len(col) % 2 == 0:
                    summary_table[i].append((col[len(col) // 2] + col[len(col) // 2 - 1]) / 2)
                else:
                    summary_table[i].append(col[len(col) // 2])

        return MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], summary_table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        new_table = []
        new_header = self.column_names.copy()
        [new_header.append(att) for att in other_table.column_names if att not in key_column_names]
        for row in self.data:
            for other_row in other_table.data:
                match = True
                new_entry = []
                for name in key_column_names:
                    if row[self.column_names.index(name)] != \
                        other_row[other_table.column_names.index(name)]:
                        match = False
                if match:
                    combined_header = self.column_names + other_table.column_names
                    combined_row = row + other_row
                    for col in new_header:
                        new_entry.append(combined_row[combined_header.index(col)])
                    new_table.append(new_entry)

        return MyPyTable(new_header, new_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        new_table = []
        new_header = self.column_names.copy()
        [new_header.append(att) for att in other_table.column_names if att not in key_column_names]
        combined_header = self.column_names + other_table.column_names

        for row in self.data:
            row_match = False
            for other_row in other_table.data:
                match = True
                for name in key_column_names:
                    if row[self.column_names.index(name)] != \
                        other_row[other_table.column_names.index(name)]:
                        match = False
                if match:
                    combined_row = row + other_row
                    new_entry = []
                    for col in new_header:
                        new_entry.append(combined_row[combined_header.index(col)])
                    new_table.append(new_entry)
                    row_match = True
            if not row_match:
                new_entry = []
                for col in new_header:
                    try:
                        new_entry.append(row[self.column_names.index(col)])
                    except ValueError:
                        new_entry.append("NA")
                new_table.append(new_entry)
        for other_row in other_table.data:
            overall_match = False
            for new_row in new_table:
                match = True
                for name in key_column_names:
                    if other_row[other_table.column_names.index(name)] != \
                        new_row[new_header.index(name)]:
                        match = False
                    if not match:
                        break
                if match:
                    overall_match = True
                    break
            if not overall_match:
                new_entry = []
                for col in new_header:
                    try:
                        new_entry.append(other_row[other_table.column_names.index(col)])
                    except ValueError:
                        new_entry.append("NA")
                new_table.append(new_entry)

        return MyPyTable(new_header, new_table)

    def get_row(self, index):
        """Gets data from a single row of the table.

        Args:
            index(int): the index of the row to get.

        Returns:
            list: a 1D list of the data in the row.
        """
        return self.data[index]

    def add_column(self, col_name, col):
        """Adds a column to the table.

        Args:
            col_name (string): the column name to add to the header.
            col (list): the data to add to each row (must have the same dimension as the table).
        """
        self.column_names.append(col_name)
        if len(col) != len(self.data):
            print("Warning: new column not the same dimension!")
            return
        for i in range(len(self.data)):
            self.data[i].append(col[i])


def remove_missing_values(list_vals):
    """Remove data from a single column (1D list) that contain a missing value ("NA").

    Args:
        col(list): the 1D list to remove missing values from.
    """
    remove_indexes = []
    for i in range(len(list_vals)):
        if list_vals[i] == "NA":
            remove_indexes.append(i)
    remove_indexes.sort(reverse=True)
    for index in remove_indexes:
        list_vals.pop(index)
        