import time

def test_function(function):
    '''
    time does a weird thing: solved in 0.0ms?
    '''

    start = time.time()
    output = function
    end = time.time()
    
    print(f"output: {output}")
    print(f"this took {(end - start) * 1000} ms")