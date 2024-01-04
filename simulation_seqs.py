import numpy as np


from hkstools.hksimulation.Simulation_Branch_HP import Simulation_Branch_HP
import ast
import argparse

np.random.seed(0)
import json


def generate(args):

    for K in [2, 3, 4, 5]:
        for mudelta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

    # nTest = args.nTest
    # nSeg = args.nSeg
    # mudelta = args.mudelta
            D = 3
            options = ast.literal_eval(args.options)
            mucenter = np.array([0.5,0.5,0.5])
            SeqsMix = []
            para={'landmark': [0]}
            L = len(para['landmark'])
            para['A'] = np.zeros((D, D, L))
            for l in range(1, L + 1):
                para['A'][:, :, l - 1] = (0.7**l) * np.random.rand(D,D)
            eigvals_list = []
            eigvecs_list = []
            for l in range(L):
                eigvals, eigvecs = np.linalg.eigh(para['A'][:, :, l])
                eigvals_list.append(eigvals)
                eigvecs_list.append(eigvecs)
            all_eigvals = np.concatenate(eigvals_list)
            max_eigval = np.max(all_eigvals)
            para['A'] = 0.005 * para['A'] / max_eigval
            for i in range(K):
                print('01 Simple exponential kernel')
                para1 = {'kernel': 'exp', 'landmark': [0]}
                para1['mu'] = mucenter+i*mudelta
                para1['A'] = para['A']
                para1['w'] = 0.5
                Seqs1 = Simulation_Branch_HP(para1, options)
                SeqsMix = SeqsMix + Seqs1
                # # 02 cluster
                # print('02 Simple exponential kernel')
                # para2 = {'kernel': 'exp', 'landmark': [0]}
                # para2['mu'] = mucenter
                # L = len(para2['landmark'])
                # para2['A'] = np.zeros((D, D, L))
                # for l in range(1, L + 1):
                #     para2['A'][:, :, l - 1] = (0.7**l) * np.random.rand(D,D)
                # eigvals_list = []
                # eigvecs_list = []
                # for l in range(L):
                #     eigvals, eigvecs = np.linalg.eigh(para2['A'][:, :, l])
                #     eigvals_list.append(eigvals)
                #     eigvecs_list.append(eigvecs)
                # all_eigvals = np.concatenate(eigvals_list)
                # max_eigval = np.max(all_eigvals)
                # para2['A'] = para1['A']
                # para2['w'] = para1['w']
                # Seqs2 = Simulation_Branch_HP(para2, options)
                # SeqsMix = Seqs1 + Seqs2
                # for seq in SeqsMix:
                #     seq['Time'] = seq['Time']
                # # print('',para1['mu'])

            ## 存储为json文件
                # 指定 JSON 文件路径
            json_file_name = 'D'+str(D)+'_'+'K'+str(K)+'_'+'mu0.5'+'mudelta'+str(mudelta)+'.json'
            json_file_path = '/media/disk1/chatgpt/aaapointprocess/mcmcbmm/seqmix_mu05_A0005/' + json_file_name
            print("saving ------")
        # 将序列写入 JSON 文件
            with open(json_file_path, 'w') as json_file:
                json.dump(SeqsMix, json_file, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def Main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--options', type=str,default="{ \
    'N': 100, 'Nmax': 100, 'Tmax': 50, 'tstep': 0.1, \
    'dt': [0.1], 'M': 250, 'GenerationNum': 10 }")

    args = parser.parse_args()

    generate(args)

if __name__ == '__main__':
    Main()