---
title: "10 Minutes to Pandas"
permalink: /pandas/
sidebar:
  - nav: docs  
---

If you are in data science, there are high chances of using pandas in your Data science and Machine Learning processes and data pipelines. Considering the need to refer to syntax and the basics of pandas, here is a quick 10-min intro to pandas and the most used methods from it.

Note: In this article, "pd" is alias for pandas and "np" is an alias for numpy.

### Object Creation

Creating a Series by passing a list of values, letting pandas create a default integer index:

```python
series = pd.Series([1,3,5,np.nan,6,8])
series

0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64

dates = pd.date_range('20130101', periods=6)
dates

DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')              

test_df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
test_df
```

| Date     |        A |         B |      C  |       D  |
|----------|----------|----------|----------|----------|
2013-01-01  | -0.165045 | 0.286237  | -0.388395 | 0.189089
2013-01-02  | -0.380108 | 0.781734  | -0.668664|  0.122847
2013-01-03  | 1.982129  | 1.970573  | 1.724951| -0.810865
2013-01-04  | -1.390268 |-0.862023  | 1.708512| -1.268239
2013-01-05  | 1.007223  | 0.024108  | 0.539417| 1.442396
2013-01-06  | 1.223380  | -0.034152 | 0.349011| -0.225668

### Viewing Data

Here is how to view the top and bottom rows of the frame.

```python
df.head()
df.tail(3)
```

| Date     |     A    |     B    |    C     |     D    |
|----------|----------|----------|----------|----------|
2013-01-03 |  1.982129  |	1.970573  |	1.724951	| -0.810865
2013-01-04 |	-1.390268 |	-0.862023 |	1.708512	| -1.268239
2013-01-05 |	1.007223  | 0.024108	| 0.539417	| 1.442396

Display the index, columns, and the underlying NumPy data:

```python
df.index

DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')

df.columns

Index(['A', 'B', 'C', 'D'], dtype='object')

df.values

array([[-0.16504516,  0.28623677, -0.38839496,  0.1890891 ],
       [-0.38010769,  0.78173448, -0.66866431,  0.12284665],
       [ 1.98212925,  1.9705729 ,  1.72495074, -0.81086545],
       [-1.39026802, -0.86202321,  1.70851228, -1.26823932],
       [ 1.0072233 ,  0.02410772,  0.53941737,  1.44239551],
       [ 1.22337986, -0.03415161,  0.34901142, -0.22566768]])
```

describe() shows a quick statistic summary of your data

```python
df.describe()
```

|     |        A |         B    |      C   |      D   |
|----------|----------|----------|----------|----------|
count |	6.000000	| 6.000000  | 6.000000  |	6.000000
mean	| 0.379552	| 0.361080  | 0.544139  |-0.091740
std	  | 1.239371	| 0.952760  | 1.012787  | 0.937839
min	  | -1.390268	| -0.862023	| -0.668664	| -1.268239
25%	  | -0.326342	| -0.019587	| -0.204043	| -0.664566
50%	  | 0.421089	| 0.155172	| 0.444214	| -0.051411
75%	  | 1.169341	| 0.657860	| 1.416239	| 0.172528
max	  | 1.982129	| 1.970573	| 1.724951	| 1.442396

Transposing your data

```python
df.T
```
2013-01-01 00:00:00	|  2013-01-02 00:00:00	 |  2013-01-03 00:00:00	 | 2013-01-04 00:00:00	|   2013-01-05 00:00:00 |  2013-01-06 00:00:00  |
|-------------------|------------------------|-----------------------|----------------------|-----------------------|-----------------------|

A	|  -0.165045  |         0.380108           |    1.982129	         |   -1.390268	        |      1.007223	        |     1.223380
B	|   0.286237	|         0.781734	         |    1.970573	         |   -0.862023	        |      0.024108	        |     -0.034152

Sorting by an axis

```python
df.sort_index(axis=1, ascending=False)
```

| Date     |    A     |    B        |    C      |     D   |
|----------|-----------|------------|---------- |----------|
2013-01-01 |	0.189089  |	-0.388395	| 0.286237  |	-0.165045
2013-01-02 |  0.122847  | -0.668664	| 0.781734  | -0.380108
2013-01-03 |	-0.810865	| 1.724951	| 1.970573  | 1.982129
2013-01-04 | 	-1.268239	| 1.708512	| -0.862023 |	-1.390268
2013-01-05 | 	1.442396	| 0.539417	| 0.024108	| 1.007223
2013-01-06 |	-0.225668	| 0.349011	| -0.034152	| 1.223380

Sorting by values

```python
df.sort_values(by='B')
```

| Date     |    A       |    B       |    C      |     D   |
|----------|----------- |------------|-----------|----------|                  
2013-01-04 |  -1.390268 |	-0.862023	 | 1.708512	 | -1.268239
2013-01-06 |  1.223380	| -0.034152	 | 0.349011	 | -0.225668
2013-01-05 |  1.007223	| 0.024108	 | 0.539417	 | 1.442396
2013-01-01 |	-0.165045 | 0.286237	 | -0.388395 | 0.189089
2013-01-02 |	-0.380108 | 0.781734	 | -0.668664 | 0.122847
2013-01-03 |  1.982129	| 1.970573	 | 1.724951	 | -0.810865


## Selection

While Standard Python / Numpy expressions for selecting and setting are intuitive and come in handy for interactive work, for production code, it is recommended to use the optimized pandas data access methods such as .at, .iat, .loc, etc..

Selecting a single column, which yields a Series, equivalent to df.A.

```python
df['A']


2013-01-01   -0.165045
2013-01-02   -0.380108
2013-01-03    1.982129
2013-01-04   -1.390268
2013-01-05    1.007223
2013-01-06    1.223380
Freq: D, Name: A, dtype: float64
```

Selecting via [], which slices the rows.

```python
df[0:3]
```

| Date     |    A       |    B       |    C      |     D   |
|----------|----------- |------------|-----------|----------|  
2013-01-01 | -0.165045	| 0.286237	 | -0.388395 |	0.189089
2013-01-02 | -0.380108	| 0.781734	 | -0.668664 |	0.122847
2013-01-03 |  1.982129	| 1.970573	 | 1.724951	 | -0.810865

Selection by Label for getting a cross section using a label

```python
df.loc[dates[0]]

A   -0.165045
B    0.286237
C   -0.388395
D    0.189089
Name: 2013-01-01 00:00:00, dtype: float64
```

Selecting on a multi-axis by label:

```python
df.loc[:,['A','B']]
```

| Date     |    A       |    B       
|----------|----------- |------------
2013-01-01 | -0.165045  |	0.286237
2013-01-02 | -0.380108	| 0.781734
2013-01-03 |	1.982129	| 1.970573
2013-01-04 |	-1.390268	| -0.862023
2013-01-05 |	1.007223	| 0.024108
2013-01-06 |	1.223380	| -0.034152

Showing label slicing, both endpoints are included:

```python
df.loc['20130102':'20130104',['A','B']]
```

|  Date     |    A       |    B       
|----------|----------- |------------
2013-01-02 |	-0.380108	| 0.781734
2013-01-03 |	1.982129	| 1.970573
2013-01-04 |	-1.390268	| -0.862023

Reduction in the dimensions of the returned object

```python
df.loc['20130102',['A','B']]

A   -0.380108
B    0.781734
Name: 2013-01-02 00:00:00, dtype: float64
```

For getting a scalar value:

```python
df.loc[dates[0],'A']
```

For getting fast access to a scalar (equivalent to the prior method):

```python
df.at[dates[0],'A']
```

## Selection by Position

Select via the position of the passed integers:

```python
df.iloc[3]

A   -1.390268
B   -0.862023
C    1.708512
D   -1.268239
Name: 2013-01-04 00:00:00, dtype: float64
```

By integer slices, acting similar to numpy/python:

```python
df.iloc[3:5,0:2]
```

|  Date    |    A      |    B       
|----------|-----------|------------
2013-01-04 | -1.390268 | -0.862023
2013-01-05 |	1.007223 |	0.024108

By lists of integer position locations, similar to the numpy/python style:

```python
df.iloc[[1,2,4],[0,2]]
```

|  Date    |        A      |    C  
|----------|---------------|------------
2013-01-02 |	  -0.380108	 |  -0.668664
2013-01-03 |	  1.982129	 |  1.724951
2013-01-05 |	  1.007223	 |  0.539417

For slicing rows explicitly:

```python
df.iloc[1:3,:]
```

|  Date    |   A       |     B      |    C       |    D   
|----------|---------  |------------|------------|---------
2013-01-02 | -0.380108 |	0.781734	| -0.668664	 | 0.122847
2013-01-03 |	1.982129 |	1.970573	| 1.724951	 |-0.810865

For slicing columns explicitly:

```python
df.iloc[:,1:3]
```

|  Date    |     A        |     B
|----------|--------------|------------
2013-01-01 |  0.286237	  |   -0.388395
2013-01-02 |	0.781734	  |   - 0.668664
2013-01-03 |	1.970573	  |   1.724951
2013-01-04 |	-0.862023	  |   1.708512
2013-01-05 |	0.024108	  |   0.539417
2013-01-06 |	-0.034152	  |   0.349011

## Boolean Indexing

Using a single column’s values to select data.

```python
df[df.A > 0]
```

|  Date    |   A       |     B      |    C       |    D   
|----------|-----------|------------|------------|---------
2013-01-03 |	1.982129 |	1.970573	|  1.724951	 | -0.810865
2013-01-05 |	1.007223 |	0.024108	|  0.539417	 | 1.442396
2013-01-06 |	1.223380 |	-0.034152	|  0.349011	 | -0.225668

Selecting values from a DataFrame where a Boolean condition is met.

```python
df[df > 0]
```

|  Date    |   A       |     B      |    C       |    D   
|----------|-----------|------------|------------|---------           
2013-01-01 |  NaN	     |   0.286237	|  NaN	     |  0.189089
2013-01-02 |  NaN	     |   0.781734	|  NaN	     |  0.122847
2013-01-03 |  1.982129 |	 1.970573 |	 1.724951	 |  NaN
2013-01-04 |  NaN	     |   NaN	    |  1.708512	 |  NaN
2013-01-05 |  1.007223 |	 0.024108	|  0.539417	 |  1.442396
2013-01-06 |  1.223380 |	 NaN	    |  0.349011	 |  NaN

Using the isin() method for filtering:

```python
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
df2
```

|  Date    |   A       |     B      |    C      |    D      |   E
|----------|-----------|------------|-----------|-----------|--------
2013-01-01 | -0.165045 |	0.286237  |	-0.388395 |	0.189089  |  one
2013-01-02 | -0.380108 |	0.781734	| -0.668664	| 0.122847  |	 one
2013-01-03 | 1.982129	 |  1.970573	| 1.724951	| -0.810865 |	 two
2013-01-04 | -1.390268 |  -0.862023	| 1.708512	| -1.268239	| three
2013-01-05 |	1.007223 |	0.024108	| 0.539417	| 1.442396	| four
2013-01-06 |	1.223380 |	-0.034152	| 0.349011	| -0.225668	| three

```python
df2[df2['E'].isin(['two','four'])]
```

|  Date    |   A       |     B      |    C      |    D      |   E
|----------|-----------|------------|-----------|-----------|--------
2013-01-03 |	1.982129 |  1.970573	| 1.724951	| -0.810865	| two
2013-01-05 |	1.007223 |	0.024108	| 0.539417	| 1.442396	| four

Setting a new column automatically aligns the data by the indexes.

```python
new_series = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
new_series 

2013-01-02    1
2013-01-03    2
2013-01-04    3
2013-01-05    4
2013-01-06    5
2013-01-07    6
Freq: D, dtype: int64
```

Setting values by label:

```python
df.at[dates[0],'A'] = 0
```

Setting by assigning with a NumPy array:

```python
df.loc[:,'D'] = np.array([5] * len(df))
```

## Missing Data

Pandas primarily uses the value np.nan to represent missing data. It is by default not included in computations. Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data.

```python
df = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df.loc[dates[0]:dates[1],'E'] = 1
df
```

|  Date    |   A        |     B      |    C       |  D   | E    |  F
|----------|------------|------------|------------|------|------|--------
2013-01-01 |  0.000000	|  0.000000	 |  -0.388395	|  5	 | NaN	|  1.0
2013-01-02 |	-0.380108	|  0.781734	 |  -0.668664	|  5	 | 1.0	|  1.0
2013-01-03 |	1.982129	|  1.970573	 |  1.724951	|  5	 | 2.0	|  NaN
2013-01-04 |	-1.390268	|  -0.862023 |	1.708512	|  5	 | 3.0	|  NaN

To drop any rows that have missing data.

```python
df.dropna(how='any')
```

|  Date    |    A       |     B     |    C      |  D    |  E   |  F
|----------|----------- |-----------|-----------|-------|------|-----                
2013-01-02 |	-0.380108	| 0.781734	| -0.668664	|  5	  | 1.0	 |  1.0

Filling missing data.

```python
df.fillna(value=5)
```

|  Date    |     A      |    B      |    C      |  D    |  E   |  F
|----------|----------- |-----------|-----------|-------|------|-----
2013-01-01 |  0.000000  |	0.000000	| -0.388395	|  5   	| 5.0	 | 1.0
2013-01-02 |	-0.380108 |	0.781734	| -0.668664	|  5	  | 1.0	 | 1.0
2013-01-03 |	1.982129	| 1.970573	| 1.724951	|  5	  | 2.0	 | 5.0
2013-01-04 |	-1.390268	| -0.862023	| 1.708512	|  5	  | 3.0	 | 5.0

To get the boolean mask where values are nan.

```python
pd.isna(df)
```

|  Date    |     A    |    B   |   C    |  D    |  E     |  F
|----------|----------|--------|--------|-------|--------|-------              
2013-01-01 |  False   | False	 | False  | False	| True	 | False
2013-01-02 |	False	  | False	 | False	| False	| False	 | False
2013-01-03 |	False	  | False	 | False	| False	| False	 | True
2013-01-04 |	False	  | False	 | False	| False	| False	 | True

##  Apply

Applying functions to the data:

```python
df.apply(np.cumsum)
```

|  Date    |     A      |     B      |   C       |  D    |  E     
|----------|------------|------------|-----------|-------|--------
2013-01-01 |	0.000000	|  0.000000	 | -0.388395 |	5	   | NaN
2013-01-02 |	-0.380108	|  0.781734	 | -1.057059 | 10	   | 1.0
2013-01-03 |	1.602022	|  2.752307	 | 0.667891	 | 15	   | 3.0


```python
df.apply(lambda x: x.max() - x.min())
```

## String Methods

Series is equipped with a set of string processing methods in the str attribute that make it easy to operate on each element of the array. Note that pattern-matching in str generally uses regular expressions by default.

```python
str_series = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
str_series

0       A
1       B
2       C
3    Aaba
4    Baca
5     NaN
6    CABA
7     dog
8     cat
dtype: object
```

```python
s.str.lower()

0       a
1       b
2       c
3    aaba
4    baca
5     NaN
6    caba
7     dog
8     cat
dtype: object
```

## Merge

Pandas provides various facilities for easily combining together Series, DataFrame, and Panel objects with various kinds of set logic for the indexes and relational algebra functionality in the case of join / merge-type operations. Concatenating pandas objects together with concat():

```python
df = pd.DataFrame(np.random.randn(10, 4))
```

break it into pieces

```python
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)
```

|     A   |     B      |     C      |    D       
|---------|------------|------------|-----------
-0.106234 |	-0.950631	 |  1.519573	|  0.097218
1.796956	| -0.450472	 |  -1.315292	| -1.099288
1.589803	| 0.774019	 |  0.009430	| -0.227336
1.153811	| 0.272446	 |  1.984570	| -0.039846
0.495798	| 0.714185	 | -1.035842	| 0.101935
0.254143	| 0.359573	 | -1.274558	| -1.978555
0.456850	| -0.094249	 | 0.665324	  | 0.226110
-0.657296	| 0.760446	 | -0.521526	| 0.392031
0.186656	| -0.131740	 | -1.404915	| 0.501818
-0.523582	| -0.876016	 |-0.004513	  | -0.509841

## JOIN

```python
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
```

| key	 | lval
|------|------
| foo	 |  1
| foo	 |  2

| key	 | rval
|------|------
| foo	 |  4
| foo	 |  5

```python
pd.merge(left, right, on='key')
```

| key	 | lval | rval
|------|------|-------
|  foo	 |   1	|  4
|  foo	 |   1	|  5
|  foo	 |   2	|  4 
|  foo	 |   2	|  5

## Append

Append rows to a dataframe. 

```python
append_df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
append_df
```

|     A      |     B      |     C      |    D       
|------------|------------|------------|-----------
|  0.310213	 |  0.511346	|  1.891497	 | 0.491886
|  -2.099571 | 	-0.477107	|  0.701392	 | 0.452229
|  -1.508507 |	0.207553	|  0.140408	 | 0.033682
|  -1.026017 |	-1.277501	|  1.755467	 | 1.056045
|  -0.890034 |	0.726291	|  -0.419684 | -1.073366
|  -0.614249 |	1.139664	|  -1.582946 | 0.661833
|  -0.010116 |	1.877924	|  -0.015332 | 1.176713
|  -0.314318 |	1.088290	|  -0.067972 | -1.759359

## Grouping 

By "group by" we are referring to a process involving one or more of the following steps:

* Splitting the data into groups based on some criteria.
* Applying a function to each group independently.
* Combining the results into a data structure.

```python
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],'C' : np.random.randn(8), 'D' : np.random.randn(8)})
```
|     A      |     B      |     C       |    D       
|------------|------------|-------------|-----------
|     foo	   |    one	    |  -0.606619  |  0.295979
|     bar	   |    one	    |  -0.015111	| -1.662742
|     foo	   |    two	    |  -0.212922	|  1.564823
|     bar	   |    three	  |   0.332831	|  0.337342
|     foo	   |    two	    |   0.235074	| -0.568002
|     bar	   |    two	    |   -0.892237	|  0.944328
|     foo	   |    one	    |    0.558490	|  0.977741
|     foo	   |    three	  |    0.517773	|  1.052036

Grouping and then applying the sum() function to the resulting groups

```python
df.groupby('A').sum()
```

|            |     C      |     D       
|------------|------------|-------------
    A        |
|  bar	     |  -0.574517	|  -0.381072
|  foo	     |  0.491797	|  3.322576

Grouping by multiple columns forms a hierarchical index, and again we can apply the sum function.

```python
df.groupby(['A','B']).sum()
```

## Reshaping

```python
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
tuples

[('bar', 'one'),
 ('bar', 'two'),
 ('baz', 'one'),
 ('baz', 'two'),
 ('foo', 'one'),
 ('foo', 'two'),
 ('qux', 'one'),
 ('qux', 'two')]
```

```python
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df_ind = df[:4]
df_ind
```
|            |    A       |     B       
|------------|------------|-------------
first	second |	
|------------|------------|-------------
bar	|  one	 | -1.863544	|  -1.071139
    |  two	 | -0.611544	|  -0.539124
baz	|  one	 | -0.438217	|   0.995510
    |  two	 |  0.481952	|  -0.009496

The stack() method “compresses” a level in the DataFrame’s columns.

```python
stacked = df_ind.stack()
stacked
```
  
|            |    first   |  second
|------------|------------|---------------
bar |   one  |      A     |    -1.863544
    |        |      B     |    -1.071139
    |   two  |      A     |    -0.611544
    |        |      B     |    -0.539124
baz |   one  |      A     |    -0.438217
    |        |      B     |    0.995510
    |   two  |      A     |    0.481952
    |        |      B     |    -0.009496
dtype: float64

With a “stacked” DataFrame or Series (having a MultiIndex as the index), the inverse operation of stack() is unstack(), which by default unstacks the last level:

```python
stacked.unstack()
```
|            |    A      |    B
|------------|-----------|---------------
first	| second		
|------------------------|---------------
  bar	| one	 | -1.863544 |	 -1.071139
      | two	 | -0.611544 |	 -0.539124
  baz	| one	 | -0.438217 |	 0.995510
      | two	 | 0.481952	 |  -0.009496

```python
stacked.unstack(1)
```
        | second	|  one	 |     two
|--------------------------------------------
   first			
--------------|-------------|-----------------  
   bar	    A	|   -1.863544	|  -0.611544
            B	|   -1.071139	|  -0.539124 
   baz	    A	|   -0.438217	|  0.481952
            B	|   0.995510	|  -0.009496

```python
stacked.unstack(0)
```

        |  first	 |  bar	    |   biz
|----------------------------------------------
  second			
----------------|------------|-----------------  
  one	      A	  |  -1.863544 |	-0.438217
            B	  |  -1.071139 |	0.995510
  two	      A	  |  -0.611544 |	0.481952
            B	  |  -0.539124 |	-0.009496

## Pivot tables

```python
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
'B' : ['A', 'B', 'C'] * 4, 'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
'D' : np.random.randn(12), 'E' : np.random.randn(12)})
```
We can produce pivot tables from this data very easily

```python
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
```

	          |   bar	     |      foo
|----------------------------------------------      
  A	  |  B	 	
|----------------------------------------------      
one	  |  A	 | -0.023021	|  -0.678341
      |  B	 | 0.095968	  |  -0.795599
      |  C	 | 0.722170	  |   0.027855
three	|  A	 | -0.529623	|   NaN
      |  B	 |  NaN	      |   1.808185
      |  C	 | -1.165231	|   NaN
two	  |  A	 |  NaN	      |   -0.097615
      |  B	 | -2.359497	|   NaN
      |  C	 |  NaN	      |  -1.002197


# Getting Data In/Out

Writing to a csv file.

```python
df.to_csv('foo.csv')
```

Reading a CSV

```python
pd.read_csv('foo.csv')
```

Writing to a HDF5 Store.

```python
df.to_hdf('foo.h5','df')
```

Reading from a HDF5 Store.

```python
pd.read_hdf('foo.h5','df')
```

Writing to an excel file.

```python
df.to_excel('foo.xlsx', sheet_name='Sheet1')
```

Reading from an excel file

```python
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
```

Pandas is indeed a powerful package to work with, especially for data engineers, scientists who work on manipulating and analysing data. With a solid grasp of Pandas, you are well-equipped to streamline your data workflow and uncover valuable insights from your data.
