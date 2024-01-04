import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.interpolate import griddata
from pp_mix.src.proto.proto import *
from pp_mix.src.proto.Params import *
from pp_mix.interface import ConditionalMCMC_gauss
from pp_mix.interface import ConditionalMCMC_hks
from pp_mix.params_helper import *
from sklearn.datasets import make_blobs  # 用于创建模拟数据

from scipy.sparse import csr_matrix
from hkstools.hksimulation.Simulation_Branch_HP import Simulation_Branch_HP
from hkstools.Initialization_Cluster_Basis import Initialization_Cluster_Basis
from hkstools.Learning_Cluster_Basis import Learning_Cluster_Basis
from hkstools.Estimate_Weight import Estimate_Weight
from hkstools.Loglike_Basis import Loglike_Basis
from hkstools.DistanceSum_MPP import DistanceSum_MPP
from hkstools.Kernel_Integration import Kernel_Integration
from hkstools.hksimulation.Kernel import Kernel
from HawkesModel import HawkesModel

import ast
import argparse
np.random.seed(0)

import json


def generate(args):
    D = args.D
    K = args.K
    mudelta = args.mudelta
    mucenter = np.array([0.5,0.5,0.5])
    json_file_name = 'D'+str(D)+'_'+'K'+str(K)+'_'+'mu0.5'+'mudelta'+str(mudelta)+'.json'
    json_file_path = '/media/disk1/chatgpt/aaapointprocess/mcmcbmm/seqmix_mu05_A005/' + json_file_name
    with open(json_file_path, 'r') as file:
        SeqsMix = json.load(file)
        for c in range(len(SeqsMix)):
            SeqsMix[c]['Time'] = np.array(SeqsMix[c]['Time'])
            SeqsMix[c]['Mark'] = np.array(SeqsMix[c]['Mark'])
            SeqsMix[c]['Mark'] = SeqsMix[c]['Mark']
    return SeqsMix
    # options = ast.literal_eval(args.options)
    # print(args)
    # D = args.D
    # K = args.K
    # nTest = args.nTest
    # nSeg = args.nSeg
    # mudelta = args.mudelta
    # mucenter = np.array([0.5,0.5,0.5])
    # SeqsMix = []
    # for i in range(K):
    #     print('01 Simple exponential kernel')
    #     para1 = {'kernel': 'exp', 'landmark': [0]}
    #     para1['mu'] = mucenter+i*mudelta
    #     print("mu is", para1['mu'])
    #     L = len(para1['landmark'])
    #     para1['A'] = np.zeros((D, D, L))
    #     for l in range(1, L + 1):
    #         para1['A'][:, :, l - 1] = (0.7**l) * np.random.rand(D,D)
    #     eigvals_list = []
    #     eigvecs_list = []
    #     for l in range(L):
    #         eigvals, eigvecs = np.linalg.eigh(para1['A'][:, :, l])
    #         eigvals_list.append(eigvals)
    #         eigvecs_list.append(eigvecs)
    #     all_eigvals = np.concatenate(eigvals_list)
    #     max_eigval = np.max(all_eigvals)
    #     para1['A'] = 0.5 * para1['A'] / max_eigval
    #     para1['w'] = 0.5
    #     Seqs1 = Simulation_Branch_HP(para1, options)
    #     SeqsMix = SeqsMix + Seqs1
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

#     ## 存储为json文件
#         # 指定 JSON 文件路径
#     json_file_name = 'D'+str(D)+'_'+'K'+str(K)+'_'+'mudelta'+str(mudelta)+'.json'
#     json_file_path = '/media/disk1/chatgpt/aaapointprocess/mcmcbmm/seqmix/' + json_file_name

# # 将序列写入 JSON 文件
#     with open(json_file_path, 'w') as json_file:
#         json.dump(SeqsMix, json_file, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

#    return SeqsMix

def Init_hakesModel_Params(SeqsMix, args):
    Tmax = ast.literal_eval(args.options)['Tmax']
    hakes_model = HawkesModel(SeqsMix,args.init_n_clus,Tmax)
    hakes_model.Initialization_Cluster_Basis()

    N = len(SeqsMix )
    D = np.zeros(N)
    for i in range(N):
        D[i] = np.max(SeqsMix[i]['Mark'])
    D = int(np.max(D))+1

    prec_params = ExponParams()
    prec_params.C = D
    prec_params.D = hakes_model.M
    prec_params.scale = 1/args.expon_lambda

    dpp_params = DPPParams()
    dpp_params.nu = args.dpp_nu
    dpp_params.rho = args.dpp_rho
    dpp_params.N = args.dpp_N

    gamma_jump_params = GammaParams()
    gamma_jump_params.alpha = args.gamma_alpha
    gamma_jump_params.beta = args.gamma_beta

    return hakes_model,prec_params,dpp_params,gamma_jump_params


def sample(SeqsMix, pp_params, prec_params, jump_params, hakes_model, args):

    dpp_sampler = ConditionalMCMC_hks(
    pp_params=pp_params, 
    prec_params=prec_params,
    jump_params=jump_params,
    init_n_clus=args.init_n_clus)
    start = time.time()

    nburn = args.nburn
    niter = args.niter
    dpp_sampler.run(nburn, niter, 5, SeqsMix, hakes_model, args)
    dpp_times = time.time() - start
    print("the elapsed time is: ",dpp_times)

def Main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--options', type=str,default="{ \
    'N': 100, 'Nmax': 100, 'Tmax': 50, 'tstep': 0.1, \
    'dt': [0.1], 'M': 250, 'GenerationNum': 10 }")
    parser.add_argument('--D', type=int, default=3)
    parser.add_argument('--K', type=int, default=2) # 生成的真实cluster_num
    parser.add_argument('--nTest', type=int, default=5)
    parser.add_argument('--nSeg', type=int, default=5)
    parser.add_argument('--mudelta', type=int, default=0.7)

    parser.add_argument('--expon_lambda', type=float, default=0.001)

    parser.add_argument('--dpp_N', type=int, default=5)
    parser.add_argument('--dpp_nu', type=float, default=20)
    parser.add_argument('--dpp_rho', type=float, default=3)

    parser.add_argument('--gamma_alpha', type=float, default=1.0)
    parser.add_argument('--gamma_beta', type=float, default=1.0)

    parser.add_argument('--init_n_clus', type=int, default=2)

    parser.add_argument('--nburn', type=int, default=0)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--thin', type=int, default=5)

    parser.add_argument('--seed', type=int,default=0)
    
    args = parser.parse_args()

    SeqsMix = generate(args)

    hakes_model, prec_params, dpp_params, gamma_params = Init_hakesModel_Params(SeqsMix, args)
    for i in [37, 3407]:
        args.seed = i
        sample(SeqsMix, dpp_params, prec_params, gamma_params, hakes_model, args)

if __name__ == '__main__':
    Main()