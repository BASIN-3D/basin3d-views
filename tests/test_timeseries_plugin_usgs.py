import json
from os.path import dirname

import datetime as dt
import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from basin3d.core.schema.enum import ResultQualityEnum, TimeFrequencyEnum
from basin3d.synthesis import register
from basin3d_views.timeseries import get_timeseries_data


def get_text(json_file_name):
    with open(os.path.join(dirname(__file__), "resources", json_file_name)) as data_file:
        return data_file.read()


def get_json(json_file_name):
    with open(os.path.join(dirname(__file__), "resources", json_file_name)) as data_file:
        return json.load(data_file)


def get_url(data, status=200):
    """
    Creates a get_url call for mocking with the specified return data
    :param data:
    :return:
    """
    return type('Dummy', (object,), {
        "json": lambda: data,
        "status_code": status,
        "url": "/testurl"})


def get_url_text(text, status=200):
    """
    Creates a get_url_text call for mocking with the specified return data
    :param text:
    :return:
    """

    return type('Dummy', (object,), {
        "text": text,
        "status_code": status,
        "url": "/testurl"})


def test_usgs_get_data(monkeypatch):
    mock_get_url = MagicMock(side_effect=list([get_url_text(get_text("usgs_get_data_rdb_09110000.rdb")),
                                               get_url(get_json("usgs_get_data_09110000.json"))]))
    from basin3d.plugins import usgs
    monkeypatch.setattr(usgs, 'get_url', mock_get_url)

    synthesizer = register(['basin3d.plugins.usgs.USGSDataSourcePlugin'])

    usgs_data = get_timeseries_data(synthesizer=synthesizer, monitoring_feature=["USGS-09110000"],
                                    observed_property=['RDC', 'WT'], start_date='2019-10-25',
                                    end_date='2019-10-28')
    usgs_df = usgs_data.data
    usgs_metadata_df = usgs_data.metadata

    # check the dataframe
    assert isinstance(usgs_df, pd.DataFrame) is True
    for column_name in list(usgs_df.columns):
        assert column_name in ['TIMESTAMP', 'USGS-09110000__RDC__MEAN', 'USGS-09110000__WT__MEAN',
                               'USGS-09110000__WT__MIN', 'USGS-09110000__WT__MAX']
    assert usgs_df.shape == (4, 5)
    data = usgs_df.get('USGS-09110000__RDC__MEAN')
    assert list(data.values) == [4.2475270499999995, 4.219210203, 4.134259662, 4.332477591]
    assert list(data.index) == [dt.datetime(2019, 10, num) for num in range(25, 29)]

    # make sure the temporary data directory is removed
    temp_dir = os.path.join(os.getcwd(), 'temp_data')
    assert os.path.isdir(temp_dir) is False

    # check the metadata store
    # Get synthesized variable field names and values
    var_metadata = usgs_metadata_df['USGS-09110000__RDC__MEAN']
    assert var_metadata['data_start'] == dt.datetime(2019, 10, 25)
    assert var_metadata['data_end'] == dt.datetime(2019, 10, 28)
    assert var_metadata['records'] == 4
    assert var_metadata['units'] == 'm^3/s'
    assert var_metadata['basin_3d_variable'] == 'RDC'
    assert var_metadata['basin_3d_variable_full_name'] == 'River Discharge'
    assert var_metadata['statistic'] == 'MEAN'
    assert var_metadata['temporal_aggregation'] == TimeFrequencyEnum.DAY
    assert var_metadata['quality'] == ResultQualityEnum.VALIDATED
    assert var_metadata['sampling_medium'] == 'WATER'
    assert var_metadata['sampling_feature_id'] == 'USGS-09110000'
    assert var_metadata['datasource'] == 'USGS'
    assert var_metadata['datasource_variable'] == '00060'

    assert usgs_metadata_df['USGS-09110000__WT__MIN']['statistic'] == 'MIN'
    assert usgs_metadata_df['USGS-09110000__WT__MAX']['statistic'] == 'MAX'


# set the following header names
mean_rdc = 'USGS-09110000__RDC__MEAN'
mean_wt = 'USGS-09110000__WT__MEAN'
min_wt = 'USGS-09110000__WT__MIN'
max_wt = 'USGS-09110000__WT__MAX'

# set short variable names for ResultQualityEnum values
VAL = ResultQualityEnum.VALIDATED
UNVAL = ResultQualityEnum.UNVALIDATED
REJECTED = ResultQualityEnum.REJECTED
EST = ResultQualityEnum.ESTIMATED
NOT_SUP = ResultQualityEnum.NOT_SUPPORTED


@pytest.mark.parametrize('query, usgs_response, expected_shape, expected_columns, expected_record_counts, expected_quality_metadata',
                         [({'statistic': ['MEAN']}, 'usgs_get_data_09110000_MEAN.json', (4, 3), [mean_rdc, mean_wt], [4, 4], [VAL, VAL]),
                          ({'statistic': ['MIN', 'MAX']}, 'usgs_get_data_09110000_MIN_MAX.json', (4, 3), [min_wt, max_wt], [4, 4], [VAL, VAL]),
                          # quality: all VAL, query VAL'
                          ({'result_quality': [VAL]}, 'usgs_get_data_09110000_VALIDATED.json', (4, 3), [mean_rdc, mean_wt], [4, 4], [VAL, VAL]),
                          # quality: all VAL, query UNVAL
                          ({'result_quality': [UNVAL]}, 'usgs_get_data_09110000_VALIDATED.json', None, None, None, None),
                          # quality: all VAL, query REJECTED (not supported)
                          ({'result_quality': [REJECTED]}, 'usgs_get_data_09110000_VALIDATED.json', None, None, None, None),
                          # quality: mix VAL-UNVAL, query VAL
                          ({'result_quality': [VAL]}, 'usgs_get_data_09110000_VALIDATED_UNVALIDATED.json', (4, 3), [mean_rdc, mean_wt], [4, 2], [VAL, VAL]),
                          # quality: VAL-UNVAL, query UNVAL
                          ({'result_quality': [UNVAL]}, 'usgs_get_data_09110000_VALIDATED_UNVALIDATED.json', (2, 2), [mean_wt], [2], [UNVAL]),
                          # quality: mix VAL-UNVAL, query VAL-UNVAL
                          ({'result_quality': [VAL, UNVAL]}, 'usgs_get_data_09110000_VALIDATED_UNVALIDATED.json', (4, 3), [mean_rdc, mean_wt], [4, 4], [VAL, f'{VAL};{UNVAL}']),
                          # quality: mix VAL-UNVAL-EST, query VAL-UNVAL
                          ({'result_quality': [VAL, UNVAL]}, 'usgs_get_data_09110000_VALIDATED_UNVALIDATED_ESTIMATED.json', (4, 5), [mean_rdc, mean_wt, min_wt, max_wt], [4, 1, 4, 4], [UNVAL, VAL, VAL, VAL]),
                          # query: mix VAL-UNVAL-EST, query EST
                          ({'result_quality': [EST]}, 'usgs_get_data_09110000_VALIDATED_UNVALIDATED_ESTIMATED.json', (2, 2), [mean_wt], [2], [EST]),
                          # query: mix VAL-UNVAL-EST, no query (includes NOT_SUPPORTED)
                          ({}, 'usgs_get_data_09110000_VALIDATED_UNVALIDATED_ESTIMATED.json', (4, 5), [mean_rdc, mean_wt, min_wt, max_wt], [4, 4, 4, 4, 4], [UNVAL, f'{VAL};{NOT_SUP};{EST}', VAL, VAL]),
                          ],
                         ids=['statistic: mean', 'statistic: min-max',
                              'quality: all VAL, query VAL', 'quality: all VAL, query UNVAL', 'quality: all VAL, query REJECTED (not supported)',
                              'quality: mix VAL-UNVAL, query VAL', 'quality: VAL-UNVAL, query UNVAL', 'quality: mix VAL-UNVAL, query VAL-UNVAL',
                              'quality: mix VAL-UNVAL-EST, query VAL-UNVAL', 'query: mix VAL-UNVAL-EST, query EST', 'query: mix VAL-UNVAL-EST, no query (includes NOT_SUPPORTED)'])
def test_usgs_get_data_with_queries(query, usgs_response, expected_shape, expected_columns, expected_record_counts,
                                    expected_quality_metadata, monkeypatch):
    get_rdb = get_url_text(get_text("usgs_get_data_rdb_09110000.rdb"))
    mock_get_url_mean = MagicMock(side_effect=list([get_rdb,
                                                    get_url(get_json(usgs_response))]))
    from basin3d.plugins import usgs
    monkeypatch.setattr(usgs, 'get_url', mock_get_url_mean)

    synthesizer = register(['basin3d.plugins.usgs.USGSDataSourcePlugin'])

    # check filtering by query
    usgs_data = get_timeseries_data(synthesizer=synthesizer, monitoring_feature=["USGS-09110000"],
                                    observed_property=['RDC', 'WT'], start_date='2019-10-25',
                                    end_date='2019-10-28', **query)

    if expected_shape is not None:
        # check the dataframe
        usgs_df = usgs_data.data
        assert isinstance(usgs_df, pd.DataFrame) is True
        expected_columns.append('TIMESTAMP')
        for column_name in list(usgs_df.columns):
            assert column_name in expected_columns
        assert usgs_df.shape == expected_shape

        # check metadata
        usgs_metadata_store = usgs_data.metadata
        # check record counts
        for idx, column_name in enumerate(expected_columns):
            if column_name == 'TIMESTAMP':
                continue
            var_metadata = usgs_metadata_store.get(column_name)
            assert var_metadata['records'] == expected_record_counts[idx]
            result_quality = var_metadata['quality']
            expected_quality = expected_quality_metadata[idx].split(';')
            assert all(qual in result_quality for qual in expected_quality) and all(
                qual in expected_quality for qual in result_quality.split(';')) is True

    else:
        assert usgs_data.data is None
        assert usgs_data.metadata is None
