import pytest
import datavis.audio_io as aio
from datetime import datetime

filename_ok_01 = '6658c4fd3657-2020-03-17T00-05-06.wav'
filename_ok_01_date = datetime(year=2020, month=3, day=17, hour=0, minute=5, second=6)

def test_extract_datetime_from_filename():
    d1 = aio.extract_datetime_from_filename(filename_ok_01)
    assert d1 == filename_ok_01_date