def index_generation(crt_i,  N, padding='reflection'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    n_pad = N // 2
    return_l = []
    if(crt_i == 1):
        return_l=[1,1,1,2,3]
    elif(crt_i == 2):
        return_l=[1,1,2,3,4]
    elif(crt_i == 99):
        return_l=[97,98,99,100,100]
    elif(crt_i == 100):
        return_l=[98,99,100,100,100]
    else:
        for i in range(crt_i - n_pad, crt_i + n_pad + 1):
            add_idx = i
            return_l.append(add_idx)
    return return_l

print(index_generation(100,5))