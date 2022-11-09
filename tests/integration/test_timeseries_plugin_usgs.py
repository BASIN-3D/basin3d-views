import datetime as dt
import os

import pandas as pd
import pytest

from basin3d.core.schema.enum import ResultQualityEnum, TimeFrequencyEnum
from basin3d.synthesis import register
from basin3d_views.timeseries import get_timeseries_data


@pytest.mark.integration
def test_usgs_get_data():
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

    # check filtering by single statistic
    usgs_data = get_timeseries_data(synthesizer=synthesizer, monitoring_feature=["USGS-09110000"],
                                    observed_property=['RDC', 'WT'], start_date='2019-10-25',
                                    end_date='2019-10-28', statistic=['MEAN'])
    usgs_df = usgs_data.data

    # check the dataframe
    assert isinstance(usgs_df, pd.DataFrame) is True
    for column_name in list(usgs_df.columns):
        assert column_name in ['TIMESTAMP', 'USGS-09110000__RDC__MEAN', 'USGS-09110000__WT__MEAN']
    assert usgs_df.shape == (4, 3)

    # check filtering by multiple statistic
    usgs_data = get_timeseries_data(synthesizer=synthesizer, monitoring_feature=["USGS-09110000"],
                                    observed_property=['RDC', 'WT'], start_date='2019-10-25',
                                    end_date='2019-10-28', statistic=['MIN', 'MAX'])
    usgs_df = usgs_data.data

    # check the dataframe
    assert isinstance(usgs_df, pd.DataFrame) is True
    for column_name in list(usgs_df.columns):
        assert column_name in ['TIMESTAMP', 'USGS-09110000__WT__MIN', 'USGS-09110000__WT__MAX']
    assert usgs_df.shape == (4, 3)

    # check filtering by quality = VALIDATED (filter by MEAN)
    usgs_data = get_timeseries_data(synthesizer=synthesizer, monitoring_feature=["USGS-09110000"],
                                    observed_property=['RDC', 'WT'], start_date='2019-10-25',
                                    end_date='2019-10-28', result_quality=[ResultQualityEnum.VALIDATED],
                                    statistic=['MEAN'])
    usgs_df = usgs_data.data

    # check the dataframe
    assert isinstance(usgs_df, pd.DataFrame) is True
    for column_name in list(usgs_df.columns):
        assert column_name in ['TIMESTAMP', 'USGS-09110000__RDC__MEAN', 'USGS-09110000__WT__MEAN']
    assert usgs_df.shape == (4, 3)

    # check filtering by quality = UNVALIDATED (filter by MEAN)
    usgs_data = get_timeseries_data(synthesizer=synthesizer, monitoring_feature=["USGS-09110000"],
                                    observed_property=['RDC', 'WT'], start_date='2019-10-25',
                                    end_date='2019-10-28', result_quality=[ResultQualityEnum.UNVALIDATED],
                                    statistic=['MEAN'])
    usgs_df = usgs_data.data

    # check there is no dataframe because no data match the query
    assert isinstance(usgs_df, pd.DataFrame) is False
