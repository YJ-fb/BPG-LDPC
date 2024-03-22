import numpy as np
import pyldpc
import time
import utils
import os

def encode_random_message(tG, snr, seed=None):
    """
    IMPORTANT: tG can be transposed coding matrix scipy.sparse.csr_matrix
    object to speed up calculations. Randomly computes a k-bits message v,
    where G's shape is (k,n). And sends it through the canal.

    Message v is passed to G: d = v,G. d is a n-vector turned into a BPSK
    modulated vector x. Then Additive White Gaussian Noise (AWGN) is added.

    SNR is the Signal-Noise Ratio: SNR = 10log(1/variance) in decibels, where
    variance is the variance of the AWGN.
    Remember:
        1. d = v.G => tG.tv = td
        2. x = BPSK(d) (or if you prefer the math: x = pow(-1,d) )
        3. y = x + AWGN(0, snr)

    ----------------------------

    Parameters:

    tG: 2D-Array (OR scipy.sparse.csr_matrix) Transpose of Coding Matrix
    obtained from CodingMatrix functions.

    snr: the Signal-Noise Ratio: SNR = 10log(1 / variance) in decibels.
        >> snr = 10log(1 / variance)

    -------------------------------

    Returns

    Tuple(v,y) (Returns v to keep track of the random message v)

    """
    rnd = np.random.RandomState(seed)
    n, k = tG.shape

    v = rnd.randint(2, size=k)

    d = utils.binaryproduct(tG, v)
    x = pow(-1, d)

    sigma = 10 ** (-snr / 20)

    e = rnd.randn(n) * sigma

    y = x + e

    return v, y


def encode(tG, v, snr, seed=None):
    """

    IMPORTANT: if H is large, tG can be transposed coding matrix
    scipy.sparse.csr_matrix object to speed up calculations.
    Codes a message v with Coding Matrix G, and sends it through a
    noisy (default) channel. Message v is passed to tG: d = tG.tv d is a
    n-vector turned into a BPSK modulated vector x. Then Additive White
    Gaussian Noise (AWGN) is added.
    SNR is the Signal-Noise Ratio: SNR = 10log(1/variance) in decibels,
    where variance is the variance of the AWGN.

        1. d = v.G (or (td = tG.tv))
        2. x = BPSK(d) (or if you prefer the math: x = pow(-1,d) )
        3. y = x + AWGN(0,snr)

    Parameters:

    tG: 2D-Array (OR scipy.sparse.csr_matrix)Transposed Coding Matrix obtained
    from coding_matrix functions. v: 1D-Array, k-vector (binary of course ..)
    SNR: Signal-Noise-Ratio: SNR = 10log(1/variance) in decibels.

    -------------------------------

    Returns y

    """
    n, k = tG.shape

    rnd = np.random.RandomState(seed)
    d = utils.binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (- snr / 20)
    #e = rnd.randn(n) * sigma
    e = rnd.random(x.shape) * sigma
    y = x + e

    return y


root_dir = 'D:\\1.安全通信\\vae-master\\image-transmission-using-bpg-ldpc-master\\Kodak24\\original\\'
for item in os.listdir(root_dir):   # 遍历root_dir
        name = root_dir + item
        save_dir = 'D:\\1.安全通信\\vae-master\\image-transmission-using-bpg-ldpc-master\\Kodak24\\encode\\'   # 存储编码结果
        save_dir1 = 'D:\\1.安全通信\\vae-master\\image-transmission-using-bpg-ldpc-master\\Kodak24\\decode\\'   # 存储解码结果
        save_dir2 = 'D:\\1.安全通信\\vae-master\\image-transmission-using-bpg-ldpc-master\\Kodak24\\ldpc_encode_data\\'  # 存储ldpc编码结果
        save_dir3 = 'D:\\1.安全通信\\vae-master\\image-transmission-using-bpg-ldpc-master\\Kodak24\\ldpc_decode_data\\'    # 存储ldpc解码结果

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)

        if not os.path.exists(save_dir2):
            os.makedirs(save_dir2)

        if not os.path.exists(save_dir3):
            os.makedirs(save_dir3)

        os.system('.\\bpgenc -m 1 -b 8 -q 50 ' + name + ' -o ' + save_dir + item.split('.')[0] + '.bin')
        print(name)

        with open(save_dir+item.split('.')[0]+'.bin', 'rb') as f:
            data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
        print(data.shape)
        n = 50
        d_v = 3
        d_c = 5
        snr = 10
        encode_start_time = time.time()
        H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
        print(H)
        print(G)
        # 将数据分成块，并按照k的大小进行划分

        # n,k = G.shape
        # n_blocks = len(data) // k
        # data_blocks = np.reshape(data[:n_blocks * k], (-1, k))
        # print(data_blocks.shape)

        # n, k = G.shape
        k = G.shape[1]
        n_blocks = len(data) // k
        remainder = len(data) % k

        # 如果有剩余数据，将其填充到长度为 k 的新数据块中
        if remainder > 0:
            padding_len = k - remainder
            last_block = np.pad(data[n_blocks * k:], (0, padding_len), mode='constant')
            data_blocks = np.vstack((data[:n_blocks * k].reshape(-1, k), last_block))
        else:
            data_blocks = data[:n_blocks * k].reshape(-1, k)
            padding_len = 0

        print(data_blocks.shape)

        # 对每个块进行LDPC编码
        # encoded_data_blocks = np.empty((n_blocks, n), dtype=np.uint8)
        # decoded_data_blocks = np.empty((n_blocks, k), dtype=np.uint8)
        # print(data_blocks[1])
        print(data_blocks.shape[1])
        Encoded_data_blocks = np.vstack([encode(G, data_blocks[i], snr) for i in range(data_blocks.shape[0])])
        # encoded_data_blocks = encode(G, data_blocks.T, snr)

        # 记录编码结束时间
        encode_end_time = time.time()

        # 计算编码耗时
        encoding_time = encode_end_time - encode_start_time
        print(f"编码耗时：{encoding_time} 秒")

        decode_start_time = time.time()
        # print(encoded_data_blocks)
        # y = pyldpc.decode(H, encoded_data_blocks, snr, maxiter=1000)
        # y = pyldpc.decode(H, encoded_data_blocks, snr)
        # y = pyldpc.decode(H, Encoded_data_blocks, snr)

        y = np.vstack([pyldpc.decode(H, Encoded_data_blocks[i], snr) for i in range(Encoded_data_blocks.shape[0])])

        # for i in range (data_blocks.shape[0]):
        #    decoded_data_blocks = pyldpc.get_message(G, y.T[i])
        #    print(decoded_data_blocks)
        # 将编码后的数据保存为二进制文件
        decoded_data_blocks_all = np.concatenate([pyldpc.get_message(G, y[i]) for i in range(data_blocks.shape[0])])

        # 记录解码结束时间
        decode_end_time = time.time()

        # 计算解码耗时
        decoding_time = decode_end_time - decode_start_time
        print(f"解码耗时：{decoding_time} 秒")
        # 将01比特流打包成uint8类型
        print(decoded_data_blocks_all)

        if padding_len > 0:
            decoded_data_blocks_all = decoded_data_blocks_all[:-padding_len]

        decoded_data_blocks_all = np.packbits(decoded_data_blocks_all)
        # 将解码后的数据写入二进制文件

        with open(save_dir3 + item.split('.')[0] + '.bin', 'wb') as f:
            decoded_data_blocks_all.tofile(f)

        os.system('.\\bpgdec -o ' + save_dir1 + item.split('.')[0] + '.png' + ' ' + save_dir3 + item.split('.')[0] + '.bin')