import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGenerator, Clients
from sympy import isprime
import random


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-np', '--num_of_participants', type=int, default=100, help='numer of the clients')
parser.add_argument('-kp', '--k_positions', type=int, default=10, help='number of positions that each participant can choose')

parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-dr', '--drop_rate', type=float, default=0.1, help='drop rate')

parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def generate_params():
    # Select a large prime number for p
    p = None
    while not p or not isprime(p):
        p = random.randint(1e9, 1e10)
    # Define a group G as a set of integers for demonstration
    G = set(range(1, p))
    # Select random elements g and h from G
    g = random.choice(list(G))
    h = random.choice(list(G))
    # Select a random element a from Z*p
    a = random.choice(list(G))

    return {"G": G, "g": g, "h": h, "p": p, "a": a}
'''
from Crypto.PublicKey import ECC
from Crypto.Random import get_random_bytes
import Crypto.Util.number as number

# 生成一个椭圆曲线密钥对，用于获取群 G 和元素 g
key = ECC.generate(curve='P-256')
G = key._curve  # 椭圆曲线群
g = key.pointQ  # 群 G 中的一个元素

# 生成素数 p 和模 p 的乘法群 Z*p 中的元素 a
p = number.getPrime(256)  # 生成一个 256 位的素数
a = number.getRandomRange(1, p)  # 获取一个小于 p 的随机数

# 将参数打包
param = {'G': G, 'g': g, 'p': p, 'a': a}
'''

if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    int Zp
    pass

    param = generate_params()
    total_positions = args['num_of_participants'] * args['k_positions']
    Clients.k_positions = args['k_positions']

    skp =
    pkp = param['g'] ** skp

    myClients = ClientsGenerator('mnist', args['IID'], args['num_of_participants'], dev)
    testDataLoader = myClients.test_data_loader
    clients_set = myClients.getClients()
    Clients.clients_set = clients_set

    Np = int(max(args['num_of_participants'] * args['cfraction'], 1)) # number in communication


    global_parameters = {}
    for key, var in net.state_dict().items(): # 将net中的参数保存在字典中（是参数，不是训练梯度）
        # key,value格式例子：
        # conv1.weight 	 torch.Size([6, 3, 5, 5])
        # conv1.bias 	 torch.Size([6])
        # conv2.weight 	 torch.Size([16, 6, 5, 5])
        # conv2.bias 	 torch.Size([16])
        # fc1.weight 	 torch.Size([120, 400])
        # fc1.bias 	     torch.Size([120])
        # fc2.weight 	 torch.Size([84, 120])
        # fc2.bias 	     torch.Size([84])

        # .state_dict() 将每一层与它的对应参数建立映射关系
        # .item() 取出tensor中的值，变为Python的数据类型
        global_parameters[key] = var.clone()  # clone原来的参数，并且支持梯度回溯

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_participants']) # Shuffle the clients
        clients_in_comm = ['client{}'.format(i) for i in order[0:Np]]
        Clients.clients_in_comm = clients_in_comm # Send to all clients

        '''=====数据位置生成阶段====='''
        Pi = clients_in_comm[0]

        clients_set[Pi].round1_firstClient(arg['k_positions'])

        '''total_public_parameters = 0
        for client in clients_in_comm[1:]:
            total_public_parameters += myClients.clients_set[client].public_paramter

        for client in tqdm(clients_in_comm):
            myClients.clients_set[client].round1_firstClient(args['num_of_participants'], arg['k_positions'], param, total_public_parameters)
'''

        sum_parameters = None
        for client in tqdm(clients_in_comm):

            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            if sum_parameters is None: # First iteration
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else: # Not first iteration
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / Np)

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_participants'],
                                                                                                args['cfraction'])))

