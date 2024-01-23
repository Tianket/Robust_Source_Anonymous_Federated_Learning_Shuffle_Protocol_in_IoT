import os
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, Clients
from sympy import isprime, nextprime
from cryptography.hazmat.primitives.asymmetric import ec


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
    curve = ec.SECP384R1()
    G = curve

    g_private_key = ec.generate_private_key(curve)
    g = g_private_key.public_key()

    h_private_key = ec.generate_private_key(curve)
    h = h_private_key.public_key()

    random_start = ec.generate_private_key(curve).private_numbers().private_value
    p = nextprime(random_start)

    a = random.randint(1, p) # 这是因为当 p 是素数时，除了 0 以外的每个数都有乘法逆元

    return {"G": G, "g": g, "h": h, "p": p, "a": a}



def elgamalEncryption(secret, pubilc_key):
    k = random.randint(1, 500)
    c1 = param['g']
    c2 = (pubilc_key ** k) * secret

    return [c1, c2]



def elgamalDecryption(messages, private_key):
    c1, c2 = messages[0], messages[1]

    return c2 * (c1 ** (-private_key))


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

    # for clients.py
    param = generate_params()
    k_positions = args['k_positions']

    private_key = random.randint(1, param['p'])
    public_key = ec.derive_private_key(private_key, param['G'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_participants'], dev)
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
        # Round 1
        Pi = clients_in_comm[0]
        myClients.round1(Pi)

        # Round 2
        for each_client in clients_in_comm:
            token, verification_information, amount_of_request_parameters = \
                myClients.clients_set[each_client].getTokenAndVerificationInformation()
            if 0:
                pass # 双线性配对函数
            else:
                random_mask = random.choice(Zp)
                # OT.Enc
                secret_list = {}
                for count in range(args['k_positions'] * Np):
                    if count == 0:
                        # C0
                        secret_list[count] = token ** random_mask
                    else:
                        Cn =
                        secret_list[count] = Cn

                myClients.clients_set[each_client].setSecretList(secret_list)

        '''=====数据匿名收集阶段====='''
        # Round 1





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

