import datetime
import os
import shutil

import pandas as pd
import pytest
from pydantic import ValidationError

from basin3d.core.schema.enum import ResultQualityEnum, TimeFrequencyEnum
from basin3d.core.types import SamplingMedium
from basin3d.synthesis import register
from basin3d_views.timeseries import TimeseriesOutputType, get_timeseries_data, PandasTimeseriesData, \
    HDFTimeseriesData


def test_get_timeseries_data_errors():
    """Test for error conditions in get timeseries data call"""
    synthesizer = register(['tests.testplugins.example.ExampleSourcePlugin'])

    # missing argument
    with pytest.raises(ValidationError):
        get_timeseries_data(synthesizer=synthesizer,
                            monitoring_features=[], observed_property_variables=['ACT'], start_date='2019-01-01')

    # missing required parameter
    with pytest.raises(ValidationError):
        get_timeseries_data(synthesizer=synthesizer,
                            observed_property_variables=['ACT'], start_date='2019-01-01')

    # output directory doesn't exist
    with pytest.raises(ValidationError):
        get_timeseries_data(synthesizer=synthesizer, output_path='./foo',
                            observed_property_variables=['ACT'], start_date='2019-01-01')


@pytest.mark.parametrize('output_type, output_path, cleanup',
                         [(TimeseriesOutputType.PANDAS, None, True),
                          (TimeseriesOutputType.HDF, None, True),
                          (TimeseriesOutputType.PANDAS, './pandas', True),
                          (TimeseriesOutputType.HDF, './hdf', True),
                          (TimeseriesOutputType.PANDAS, None, False),
                          (TimeseriesOutputType.HDF, None, False),
                          (TimeseriesOutputType.PANDAS, './pandas', False),
                          (TimeseriesOutputType.HDF, './hdf', False)],
                         ids=['pandas-cleanup', 'hdf-cleanup', 'pandas-output-cleanup',
                              'hdf-output-cleanup', 'pandas', 'hdf', 'pandas-output',
                              'hdf-output'])
def test_get_timeseries_data(output_type, output_path, cleanup):
    """Test processing for get_timeseries_data basic functionality"""

    # Create temporary directory
    if output_path:
        os.mkdir(output_path)
    try:
        synthesizer = register(['tests.testplugins.example.ExampleSourcePlugin'])

        result = get_timeseries_data(synthesizer=synthesizer, output_path=output_path, output_type=output_type,
                                     cleanup=cleanup,
                                     monitoring_features=['E-1', 'E-2', 'E-3', 'E-4'],
                                     observed_property_variables=['ACT', 'Al'], start_date='2016-02-01')

        assert result
        assert output_type is TimeseriesOutputType.PANDAS and isinstance(result, PandasTimeseriesData) or \
               output_type is TimeseriesOutputType.HDF and isinstance(result, HDFTimeseriesData)

        df = result.data
        metadata_df = result.metadata
        metadata_nodata_df = result.metadata_no_observations

        if isinstance(result, PandasTimeseriesData):
            assert isinstance(df, pd.DataFrame) is True

        else:
            assert isinstance(df, object) is True

            # Since this is an HDF file, the output directory is not remove
            assert result.output_path is not None

            assert result.hdf.attrs['aggregation_duration'] == 'DAY'
            assert list(result.hdf.attrs['monitoring_features']) == ['E-1', 'E-2', 'E-3', 'E-4']
            assert list(result.hdf.attrs['observed_property_variables']) == ['ACT', 'Al']
            assert result.hdf.attrs['query_start_time']
            assert result.hdf.attrs['query_end_time']
            assert result.hdf.attrs['start_date']
            assert list(result.hdf.attrs['variables_data']) == ['E-1__ACT__MEAN', 'E-2__ACT__MAX', 'E-4__Al__MAX']
            assert list(result.hdf.attrs['variables_nodata']) == ['E-3__Al__MEAN']

        # Check the output path
        if not cleanup:
            # No files should have been cleaned up
            assert result.output_path and os.path.exists(result.output_path)
        elif isinstance(result, PandasTimeseriesData):
            # if cleanup is true and the output is pandas,
            #  there should not be an output path
            assert result.output_path is None
        elif output_path:
            # If there is an output path, it should exist
            assert result.output_path and os.path.exists(result.output_path)

        # check the dataframe
        assert list(df.columns) == ['TIMESTAMP', 'E-1__ACT__MEAN', 'E-2__ACT__MAX', 'E-4__Al__MAX']
        assert df.shape == (9, 4)
        data = df.get('E-1__ACT__MEAN')
        assert list(data.values) == [num * 0.3454 for num in range(1, 10)]
        assert list(data.index) == [datetime.datetime(2016, 2, num) for num in range(1, 10)]

        # check the metadata with observations
        # Get synthesized variable field names and values
        var_metadata = metadata_df['E-1__ACT__MEAN']
        assert var_metadata['data_start'] == datetime.datetime(2016, 2, 1)
        assert var_metadata['data_end'] == datetime.datetime(2016, 2, 9)
        assert var_metadata['records'] == 9
        assert var_metadata['units'] == 'nm'
        assert var_metadata['basin_3d_variable'] == 'ACT'
        assert var_metadata['basin_3d_variable_full_name'] == 'Acetate (CH3COO)'
        assert var_metadata['statistic'] == 'MEAN'
        assert var_metadata['temporal_aggregation'] == TimeFrequencyEnum.DAY
        assert var_metadata['quality'] == ResultQualityEnum.VALIDATED
        assert var_metadata['sampling_medium'] == SamplingMedium.WATER
        assert var_metadata['sampling_feature_id'] == 'E-1'
        assert var_metadata['datasource'] == 'Example'
        assert var_metadata['datasource_variable'] == 'Acetate'

        assert 'E-3__Al__MEAN' in result.variables_no_observations
        assert 'E-3__Al__MEAN' not in result.variables

        assert list(metadata_df.columns) == ['TIMESTAMP', 'E-1__ACT__MEAN', 'E-2__ACT__MAX', 'E-4__Al__MAX']
        assert len(metadata_df) == 20

        # Check the other data quality aggregations
        assert metadata_df['E-2__ACT__MAX']['quality'] == ';'.join(
            [ResultQualityEnum.UNVALIDATED, ResultQualityEnum.REJECTED])
        assert metadata_df['E-4__Al__MAX']['quality'] == ';'.join(
            [ResultQualityEnum.VALIDATED, ResultQualityEnum.UNVALIDATED, ResultQualityEnum.REJECTED])

        # check the metadata with no observations
        # Get synthesized variable field names and values
        var_metadata = metadata_nodata_df['E-3__Al__MEAN']
        assert var_metadata['data_start'] is None
        assert var_metadata['data_end'] is None
        assert var_metadata['records'] == 0
        assert var_metadata['units'] == 'mg/L'
        assert var_metadata['basin_3d_variable'] == 'Al'
        assert var_metadata['basin_3d_variable_full_name'] == 'Aluminum (Al)'
        assert var_metadata['statistic'] == 'MEAN'
        assert var_metadata['temporal_aggregation'] == TimeFrequencyEnum.DAY
        assert var_metadata['quality'] is None
        assert var_metadata['sampling_medium'] == SamplingMedium.WATER
        assert var_metadata['sampling_feature_id'] == 'E-3'
        assert var_metadata['datasource'] == 'Example'
        assert var_metadata['datasource_variable'] == 'Aluminum'

        assert list(metadata_nodata_df.columns) == ['TIMESTAMP', 'E-3__Al__MEAN']
        assert len(metadata_nodata_df) == 20

    finally:
        # remove temporary directory
        if output_path and os.path.exists(output_path):
            shutil.rmtree(output_path)


VAL = ResultQualityEnum.VALIDATED
UNVAL = ResultQualityEnum.UNVALIDATED
EST = ResultQualityEnum.ESTIMATED
REJ = ResultQualityEnum.REJECTED


@pytest.mark.parametrize('filters, expected_results',
                         [
                             # monitoring_features
                             ({'monitoring_features': ['E-1', 'E-2']},
                              {'has_data': True, 'columns': ['TIMESTAMP', 'E-1__ACT__MEAN', 'E-2__ACT__MAX'],
                               'df_shape': (9, 3),
                               'no_observations_variable': None,
                               'quality_filter_checks': {}}),
                             # statistic
                             ({'monitoring_features': ['E-1', 'E-2', 'E-3', 'E-4'], 'statistic': ['MEAN']},
                              {'has_data': True, 'columns': ['TIMESTAMP', 'E-1__ACT__MEAN'], 'df_shape': (9, 2),
                               'no_observations_variable': ['E-3__Al__MEAN'],
                               'quality_filter_checks': {}}),
                             # monitoring_feature_and_statistic
                             ({'monitoring_features': ['E-3', 'E-4'], 'statistic': ['MIN']},
                              {'has_data': False, 'columns': None, 'df_shape': None,
                               'no_observations_variable': None,
                               'quality_filter_checks': {}}),
                             # quality-VALIDATED
                             ({'monitoring_features': ['E-1', 'E-2', 'E-3', 'E-4'], 'result_quality': [VAL]},
                              {'has_data': True, 'columns': ['TIMESTAMP', 'E-1__ACT__MEAN', 'E-4__Al__MAX'],
                               'df_shape': (9, 3),
                               'no_observations_variable': [],
                               'quality_filter_checks': [('E-1__ACT__MEAN', 9, [VAL]), ('E-4__Al__MAX', 7, [VAL])]}),
                             # quality-UNVALIDATED
                             ({'monitoring_features': ['E-1', 'E-2', 'E-3', 'E-4'], 'result_quality': [UNVAL]},
                              {'has_data': True, 'columns': ['TIMESTAMP', 'E-2__ACT__MAX', 'E-4__Al__MAX'],
                               'df_shape': (7, 3),
                               'no_observations_variable': [],
                               'quality_filter_checks': [('E-2__ACT__MAX', 7, [UNVAL]), ('E-4__Al__MAX', 1, [UNVAL])]}),
                             # quality-VALIDATED+UNVALIDATED
                             ({'monitoring_features': ['E-1', 'E-2', 'E-3', 'E-4'], 'result_quality': [VAL, UNVAL]},
                              {'has_data': True,
                               'columns': ['TIMESTAMP', 'E-1__ACT__MEAN', 'E-2__ACT__MAX', 'E-4__Al__MAX'],
                               'df_shape': (9, 4),
                               'no_observations_variable': [],
                               'quality_filter_checks': [('E-1__ACT__MEAN', 9, [VAL]), ('E-2__ACT__MAX', 7, [UNVAL]),
                                                         ('E-4__Al__MAX', 8, [VAL, UNVAL])]}),
                             # quality-ESTIMATED: no data
                             ({'monitoring_features': ['E-1', 'E-2', 'E-3', 'E-4'], 'result_quality': [EST]},
                              {'has_data': False, 'columns': None, 'df_shape': None,
                               'no_observations_variable': [],  # b/c no MeasTVPOvs returned, this is none
                               'quality_filter_checks': []}),
                             # statistic_and_quality: MAX and VALIDATED
                             ({'monitoring_features': ['E-1', 'E-2', 'E-3', 'E-4'], 'result_quality': [VAL],
                               'statistic': ['MEAN']},
                              {'has_data': True, 'columns': ['TIMESTAMP', 'E-1__ACT__MEAN'], 'df_shape': (9, 2),
                               'no_observations_variable': [],
                               'quality_filter_checks': [('E-1__ACT__MEAN', 9, [VAL])]}),
                         ],
                         ids=['monitoring_features', 'statistic', 'monitoring_feature_and_statistic',
                              'quality-VALIDATED', 'quality-UNVALIDATED', 'quality-VALIDATED+UNVALIDATED',
                              'quality-ESTIMATED',
                              'statistic_and_quality: MAX and VALIDATED'])
def test_get_timeseries_data_filtering(filters, expected_results):
    """Test processing for get_timeseries_data statistic"""

    synthesizer = register(['tests.testplugins.example.ExampleSourcePlugin'])

    result = get_timeseries_data(synthesizer=synthesizer, output_path=None,
                                 output_type=TimeseriesOutputType.PANDAS, cleanup=True,
                                 observed_property_variables=['ACT', 'Al'], start_date='2016-02-01',
                                 **filters)
    assert result
    df = result.data
    if expected_results['has_data']:
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == expected_results['columns']
        assert df.shape == expected_results['df_shape']
    else:
        assert df is None

    no_observation_variable = expected_results['no_observations_variable']
    if no_observation_variable:
        for var in no_observation_variable:
            assert var in result.variables_no_observations
            assert var not in result.variables
    else:
        assert len(result.variables_no_observations) == 0

    if expected_results['quality_filter_checks']:
        for expected_result in expected_results['quality_filter_checks']:
            column_name, expected_data_values, expected_quality_metadata = expected_result
            data_count = 0
            for data_point in df.get(column_name):
                if not pd.isna(data_point):
                    data_count += 1
            metadata_store_column = result.metadata.get(column_name)
            assert data_count == expected_data_values
            assert metadata_store_column.get('records') == expected_data_values
            metadata_quality = metadata_store_column.get('quality')
            assert all(qual in metadata_quality for qual in expected_quality_metadata) and all(
                qual in expected_quality_metadata for qual in metadata_quality.split(';')) is True
