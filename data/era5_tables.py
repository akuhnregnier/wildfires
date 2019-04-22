#!/usr/bin/env python3
# -*- coding: utf-8 -*_
"""Determining the available datasets by parsing the documentation website.

"""
import logging
import logging.config
import re

from bs4 import BeautifulSoup
from joblib import Memory
import requests

from wildfires.logging_config import LOGGING


logger = logging.getLogger(__name__)
URL = "https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation"
location = "./.cachedir"
memory = Memory(location, verbose=0)


@memory.cache
def load_era5_tables(url=URL):
    """Parse ERA5 data documentation website for tables of variable names.

    Note:
        Since this function is cached using joblib.Memory.cache, a new
        output will only be computed for a given url if this function has
        never been called with this url. To execute this function again,
        delete the cache directory './cachedir'.

    Args:
        url (str): The url of the data documentation.

    Returns:
        tables (dict): Dictionary containing the table information.
            tables['header_row'] contains the table header row as a list of
                strings,
            tables['rows'] contains a list of lists which
                make up the main table content, and
            tables['caption'] contains the table caption.
        common_table_header: Since all the table headers are the same, the
            common header row is returned as a list of strings.

    """
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    # Only interested in tables 1 - 13.
    table_pattern = re.compile(r"(Table \b(?:[1-9]|1[0-3])):")

    def target_table(tag):
        # Test if tag if True first to avoid performing operations on it if
        # it is None (or '') which could throw an error.
        return (
            tag
            and "table-wrap" in tag.get("class", [None])
            and table_pattern.search(tag.previous_element)
        )

    def div_header_tags(tag):
        return (
            tag
            and tag.name == "div"
            and tag.has_attr("class")
            and "tablesorter-header-inner" in tag.get("class")
            and tag.parent.parent.parent.has_attr("class")
            and ("tableFloatingHeaderOriginal" in tag.parent.parent.parent.get("class"))
        )

    def th_header_tags(tag):
        return (
            tag
            and tag.name == "th"
            and tag.has_attr("class")
            and "confluenceTh" in tag.get("class")
        )

    # Depending on how the html source code is downloaded, the tags differ.
    header_funcs = (div_header_tags, th_header_tags)

    tables = {}
    found_table_tags = soup.find_all(target_table)
    logger.debug("Found {} table tags.".format(len(found_table_tags)))
    for table_tag in found_table_tags:
        search_result = table_pattern.search(table_tag.previous_element)
        assert search_result, (
            "target_table function should only discover tables matching "
            "table_pattern."
        )

        table_name = search_result.group(1)

        rows = []
        row_tags = table_tag.find_all("tr")
        logger.debug(
            "Found {} rows for table {} (might include header).".format(
                len(row_tags), table_name
            )
        )

        header_tags = max(
            [table_tag.find_all(header_func) for header_func in header_funcs], key=len
        )
        logger.debug(
            "Found {} header tags for table {}.".format(len(header_tags), table_name)
        )
        for row_tag in row_tags:
            entries = row_tag.find_all("p")
            td_entries = row_tag.find_all("td")
            entries = max((entries, td_entries), key=len)
            row_contents = [entry.get_text().replace("\xa0", "") for entry in entries]
            if row_contents:
                rows.append(row_contents)

        if header_tags:
            header = [tag.get_text() for tag in header_tags]
        else:
            header = rows[0]
            rows = rows[1:]

        # Process rows
        try:
            for row_index in range(len(rows)):
                rows[row_index][0] = int(rows[row_index][0])
        except ValueError:
            logger.exception("First column should be an integer index!")
            raise

        tables[table_name] = {
            "header_row": header,
            "rows": rows,
            "caption": str(table_tag.previous_element),
        }

    headers = []
    for data in tables.values():
        headers.append(tuple(data["header_row"]))

    assert (
        len(set(headers)) == 1
    ), "There should be a single common header for the selected Tables."

    logger.info("Found {} tables.".format(len(tables)))
    common_table_header = headers[0]

    return tables, common_table_header


def get_short_to_long(url=URL):
    """Get a mapping of short variable names to long variable names.

    The mapping is derived from the tables of variables found at the given
    url, which should point to the ERA5 data documentation.

    Args:
        url (str): The url of the data documentation.

    Returns:
        short_to_long (dict): Dictionary with short variable names as keys
            and long variable names as the corresponding values.

    """
    short_to_long = dict()
    tables, common_table_header = load_era5_tables(url)
    long_name_col = common_table_header.index("name")
    short_name_col = common_table_header.index("shortName")
    for data in tables.values():
        for row in data["rows"]:
            short_to_long[row[short_name_col]] = row[long_name_col]
    return short_to_long


def get_table_dict(url=URL):
    """Get a mapping of long variable names to all other properties.

    The mapping is derived from the tables of variables found at the given
    url, which should point to the ERA5 data documentation.

    Args:
        url (str): The url of the data documentation.

    Returns:
        table_dict (dict): Dictionary with long variable names as keys
            and a dictionary of properties as the corresponding values.

    """
    table_dict = dict()
    tables, common_table_header = load_era5_tables(url)
    long_name_col = common_table_header.index("name")
    for data in tables.values():
        for row in data["rows"]:
            long_name = row[long_name_col]
            table_dict[long_name] = {"caption": data["caption"]}
            for col, entry in enumerate(row):
                if col != long_name_col:
                    table_dict[long_name][common_table_header[col]] = entry

    return table_dict


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING)
    tables, common_table_header = load_era5_tables()
    short_to_long = get_short_to_long()
    table_dict = get_table_dict()
