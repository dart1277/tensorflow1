import numpy as np
import pandas as pd
import pytz


def test1():
    # pd.read_csv('a.csv', index_col='Date', parse_dates=True)
    arr = pd.array([1,2,3,4], dtype='int32')
    print(arr)
    ts00 = pd.Timestamp("2023-03-26 2:55").tz_localize(tz="CET", ambiguous=True, nonexistent='shift_forward')
    ts0 = pd.Timestamp("2023-10-29 3:00").tz_localize(tz="CET", ambiguous=True, nonexistent='shift_backward')
    ts1 =  ts0 - pd.Timedelta(minutes=1)
    ts2 = pd.Timestamp("2023-01-17", tz="UTC")
    # In addition, if you perform date arithmetic on local times that cross DST boundaries,
    # the results may be in an incorrect timezone (ie. subtract 1 minute from 2002-10-27 1:00 EST and you get 2002-10-27 0:59 EST
    # instead of the correct 2002-10-27 1:59 EDT). A normalize() method is provided to correct this. Unfortunately these issues cannot be
    # resolved without modifying the Python datetime implementation.
    print(ts00)
    print(ts0)
    print(ts1.dst())
    print(ts1) # Normalize Timestamp to midnight, preserving tz information.
    print(ts2.days_in_month)
    print(ts2.dayofyear)
    print(ts2.is_month_end)
    print(ts2.is_year_start)
    print(ts2.is_leap_year)
    td = ts2 - ts1
    td2 = pd.Timedelta(value=1, unit='days')
    print(td2.seconds)
    print(td2.components)
    print(td2.total_seconds())
    print(ts1.strftime("%Y-%m-%d %H:%M:%S"))
    ts3 = ts1.replace(year=2000)
    print(ts3.strftime("%Y-%m-%d %H:%M:%S"))
    ts4 = pd.Timestamp("2023-01-16").replace(year=2000).tz_localize(tz=pytz.timezone(pytz.country_timezones['CN'][0]))
    print(ts4.strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    ts5 = ts4.replace(year=2000).tz_convert(pytz.country_timezones['PL'][0])
    print(ts5.strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    in1 = pd.Interval(left=0, right=10, closed='left') # default is right # can be also used with dates
    print(in1)
    print(5 in in1)
    print(in1+5)
    # extend interval
    print(in1*5)
    print(in1.overlaps(in1+5))

    print(pd.IntervalIndex.from_breaks([0, 0.5, 1.0, 1.5, 2.0], closed='right'))
    print(pd.PeriodIndex(['2019', '2020'], freq="M")) # PeriodIndex(['2019-01', '2020-01'], dtype='period[M]')
    ii1 = pd.IntervalIndex.from_arrays([0, 0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0, 2.5], closed='right')
    print(ii1)
    print(ii1.contains(0.5))

    # pandas series
    s1 = pd.Series([1, 2, 3, 1, 2, 2])
    print(s1)
    #categories
    c1 = pd.Categorical(['m', 'f', 'm', np.nan])
    print(c1) # nan values are ignored
    print(pd.Categorical(['m', 'f', 'm']).codes)
    print(pd.Categorical(['m', 'f', 'm']).categories)
    print(pd.Categorical(['m', 'f', 'm'], categories=['m'], ordered=True)) # others will be nan, orederd allows sorting nad filtering

    print(c1.remove_categories(removals=['m']))

    print(s1.astype('category'))

    # sparse arrays
    arr = np.random.rand(10)
    arr[2:5] = np.nan
    sparse_arr = pd.arrays.SparseArray(arr)
    from sys import getsizeof
    print("size")
    print(getsizeof(sparse_arr))
    print(np.asarray(sparse_arr))
    print(sparse_arr.dtype) # return data type and fill value
    # sdf = data.astype(pd.SparseDtype("float", np.nan))
    # print(df.memory_usage().sum() /1e3)

    ss1 = pd.Series(["a", "aa", "aaa", " bbbbb", np.nan], dtype="string")
    print(ss1.str.lower())
    print(ss1.str.len())
    print(ss1.str.strip())
    print(ss1.str.replace("b", "_"))
    print(ss1.str.split("b"))
    ss2 = pd.Series(["a", "a a", "a aa", "b bb bbb ", np.nan], dtype="string")
    print(ss2.str.split(" ", expand=True))

    ss3 = pd.Series(["a", "aa", "aaa", " bbbbb", np.nan], dtype="string")

    print(ss3.str.cat(sep="__", na_rep="-"))
    print("cat series") # works like "zip" on strings
    print(ss3.str.cat(ss1))

    # shift data rows (indexed by date), DatetimeIndex or TimedelatIndex can also be used to create a searchable index on a data frame
    # df.shift(2)
    # df.asfreq(freq='Q').sum()

    # multi indexing
    # pd.MultiIndex.from_tuples(tuples=t, names=('country', 'year'))
    # df.loc[('China', slice(None)), :]
    # df.xs[('China', level='country'] # use cross-sections to get data from multi index data frame
    ...


if __name__ == '__main__':
    test1()
