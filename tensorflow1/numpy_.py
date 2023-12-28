import numpy as np


def test1():
    # numpy performs better than pandas in arrays in sie less tha 50k rows
    # pandas is faster for arrays of 500k rows or more
    arr = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype='float32')
    arr2 = np.array([[10], [11]], dtype='float32')
    # array broadcast
    # np datatypes: DateTime(M), Timedelta(m), Object(o), unicode string (u), void(V), structural (contains multiple datatypes)
    print(arr + arr2)
    print(arr.astype(dtype="U"))  # data type conversion
    print(np.float32(arr))  # data type conversion
    print(np.array(['2023-12-02', '2023-12-03'], dtype="M").dtype)
    print(arr.ndim)
    print(arr.nbytes)
    print(arr.itemsize)
    print(arr.shape)
    print(arr.T)
    print(np.empty((3, 4), dtype=np.float64))
    print(np.ones((1, 2)))
    print(np.zeros((1, 2), order='C'))  # C - row, F (Fortran) - col
    full = np.full((1, 2), 5)  # fill array with value of 5
    print(np.pi)
    print(np.e)

    print(full)
    print(np.asarray(full, dtype='U'))
    print(np.zeros_like(full))
    print(np.fromiter(range(10), dtype=np.float32))

    print(np.frombuffer(b'2113431221134312', dtype=int))
    print(np.arange(1, 8, 2))  # like range
    print(
        np.linspace(1, 10, num=5, dtype=float, retstep=True, endpoint=True))  # create array with evenly spaced numbers
    print(np.logspace((1, 1, 1), (10, 10, 10), num=4, base=2, dtype=float, endpoint=True))

    print(np.eye(5, 4, k=-1))  # identity matrix
    diag = np.diag([2, 3, 4, 5])
    print(diag)  # diagonal matrix
    print(np.diag(diag))  # extract diagonal

    rand_arr = np.random.rand(3, 4)
    print(rand_arr)  # random values in a given shape
    print(np.random.randn(3, 4))  # normal distribution random values in a given shape

    slice_arr = np.array([[i + j * 10 for j in range(5)] for i in range(4)])
    print(slice_arr)

    print("basic indexing")
    print(slice_arr[1])  # simple indexing returns view of array, returned by reference
    print(slice_arr[1, 1])
    print(slice_arr[1:3, 1:3])  # array slicing

    print("advanced indexing")
    # advanced integer indexing, use list for each dimension, returns copy instead of view
    print(slice_arr[[1, 2]])  # row 1, 2
    print(slice_arr[[1, 2], [1, 3]])
    print(slice_arr[[1, 2], 1])

    print("binary indexing")
    # mask and array must be the same size
    mask = np.diag([1 for _ in range(4)]).astype(dtype=bool)
    print(mask)
    arr = np.asarray(slice_arr)
    arr = arr[:4, :4]
    print(arr)
    print(arr[mask])
    print(arr[arr >= 20])
    # or
    indexer = np.where(arr >= 20)
    print(arr[indexer])
    print(slice_arr.reshape((2, 2, 5)))  # can also pass -1 as one of the dimensions
    print(slice_arr.reshape(-1))  # flatten array
    slice_copy = slice_arr.copy()
    deleted = np.delete(slice_copy, [1, 3], axis=0)  # if axis is none, out is a flattened array
    print(deleted)

    # broadcasting
    # matching dimensions are equal in size or one of dimensions is one
    print(slice_arr + np.array([_ for _ in range(5)]))

    # nditer operator object
    d3 = slice_arr.reshape((2, 2, 5))
    for _ in np.nditer(d3, order='c', op_flags=['readwrite']):  # nditer can itarate over multiple arrays [a1, a2]
        print(f"{_}")

    # copy np array
    print(slice_arr.copy())

    print(np.isnan(slice_arr).any())  # isnan returns array of booleans
    print(np.isnan(slice_arr).any(axis=1))  # isnan returns array of booleans, axis=1 returns columns for any

    # arithmetics
    print(slice_arr.sum())
    print(np.cos(slice_arr))
    print(np.abs(slice_arr).mean())
    print(np.linalg.inv(slice_arr[:4, :4]).dot(slice_arr[:4, :4]))  # svd is pseudoinverse

    # sorting arrays
    print(np.sort(slice_arr))  # returns sorted copy of array
    # slice_arr.sort() # sorts array in place
    print(np.argsort(slice_arr))  # returns sorted index positions (on axis=0)

    print("find matrix inverse")
    a = np.array([[1., 2.], [3., 4.]])
    inv = np.linalg.inv(a)
    print(inv)
    print(np.matmul(a, inv))

    ...


if __name__ == '__main__':
    test1()
