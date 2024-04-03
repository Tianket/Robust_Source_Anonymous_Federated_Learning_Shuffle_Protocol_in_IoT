import os
import sys
import argparse
import copy
from tqdm import tqdm
from decimal import getcontext, Decimal
import random
import secrets
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from sympy import isprime, nextprime
from Models import Mnist_2NN, Mnist_CNN

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-np', '--num_of_participants', type=int, default=20, help='numer of the clients')
parser.add_argument('-kp', '--k_positions', type=int, default=2, help='number of positions that each participant can choose')

parser.add_argument('-cf', '--cfraction', type=float, default=0.9, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.005, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-dr', '--drop_rate', type=float, default=0.3, help='drop rate')
parser.add_argument('-t', '--threshold', type=float, default=5, help='the minimum number of hosts that can complete the iteration')

parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')



def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def generate_params():
    binary_operator = "+"
    #binary_operator = "*"

    random_bytes = secrets.token_bytes(6)
    random_number = int.from_bytes(random_bytes, byteorder='big')
    p = nextprime(random_number)  # a large prime number

    #a = random.randint(1, p)
    a = random.randint(1, int(str(p)[:4]))

    g = random.randint(2, 5)  # generator

    if binary_operator == '+':
        G = list(set(range(0, 100, g)))
    if binary_operator == '*':
        G = [g**i for i in range(9)]

    h = random.choice(G)

    return {"G": G, "g": g, "h": h, "p": p, "a": a, "b": binary_operator}

def simulate_offline(all_clients_in_comm, drop_rate):
    random.shuffle(all_clients_in_comm)
    num_to_remove = int(len(all_clients_in_comm) * drop_rate)
    shuffled_and_removed = all_clients_in_comm[:-num_to_remove]

    return shuffled_and_removed


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #dev = torch.device("mps") if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else dev


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

    Np = int(max(args['num_of_participants'] * args['cfraction'], 1))  # number in communication
    data_positions = list(range(1, args['k_positions'] * Np + 1))
    random.shuffle(data_positions)

    # private_key = random.randint(1, int(str(param['p'])[:4]))
    # if param["b"] == "+":
    #     public_key = param['g'] * private_key
    # elif param["b"] == "*":
    #     public_key = param['g'] ** private_key

    from clients import ClientsGroup, Clients, bilinear_pairing_function, Power
    Clients.param = param
    Clients.k_positions = args['k_positions']

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_participants'], dev)
    testDataLoader = myClients.test_data_loader
    clients_set = myClients.get_clients()
    Clients.clients_set = clients_set

    global_parameters = {}
    for key, var in net.state_dict().items(): # 将net中的参数保存在字典中（是参数，不是训练梯度）
        global_parameters[key] = var.clone()

    for comm_round in range(args['num_comm']):
        print("Communicate round {} ".format(comm_round + 1), end="")

        order = np.random.permutation(args['num_of_participants']) # Shuffle the clients
        clients_in_comm = ['client{}'.format(comm_round + 1) for comm_round in order[0:Np]]
        Clients.clients_in_comm = clients_in_comm # Send to all clients

        '''=====数据位置生成阶段====='''
        # Round 1
        Pi = clients_in_comm[0]
        myClients.round1(Pi)

        '''for each_client in clients_in_comm:
            print(myClients.clients_set[each_client].request_parameters)
'''
        # Round 2
        for each_client in clients_in_comm:
            token, verification_information, amount_of_request_parameters = \
                myClients.clients_set[each_client].get_token_and_verification_information()

            k_plus_Np = args['k_positions'] * len(clients_in_comm)
            temp_exponent = param['a'] * (k_plus_Np - len(myClients.clients_set[each_client].request_parameters))

            left_side = bilinear_pairing_function(token, verification_information)

            if param["b"] == "+":
                right_side = bilinear_pairing_function(param['g'], param['h'] * temp_exponent)
            elif param["b"] == "*":
                right_side = bilinear_pairing_function(Power(param['g'], 1), Power(param['h'], temp_exponent))

            if left_side != right_side: # 双线性配对函数 bilinear pairing function
                print("===== Agreement terminated 1=====")
                sys.exit(1)
            else:
                #random_mask = random.randint(1, param['p'])
                random_mask = random.randint(1, int(str(param['p'])[:4]))
                # OT.Enc
                secret_list = []
                for count in range(args['k_positions'] * Np + 1): # count - 1 == n
                    if count == 0:
                        # C0
                        if param["b"] == "+":
                            secret_list.append(token * random_mask)
                        elif param["b"] == "*":
                            secret_list.append(token ** random_mask)
                    else:
                        # Cn
                        if param["b"] == "+":
                            secret_list.append(bilinear_pairing_function(param['g'] * (1 / (param['a'] + count)),
                                                                         param['h'] * random_mask) * data_positions[count - 1])
                        elif param["b"] == "*":
                            secret_list.append([bilinear_pairing_function(Power(param['g'], (1 / (param['a'] + count))),
                                                                         Power(param['h'], random_mask)), data_positions[count - 1]])

                myClients.clients_set[each_client].set_secret_list(secret_list)
                myClients.clients_set[each_client].decrypt_secret()

        '''=====数据匿名收集阶段====='''
        # Round 1
        u1 = clients_in_comm
        random.shuffle(u1)
        for client in u1:
            local_parameters = myClients.clients_set[client].local_update(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)

            myClients.clients_set[client].generate_anonymous_model_upload_list(global_parameters, local_parameters)
            myClients.clients_set[client].generate_and_encrypt_shared_values(args['threshold'])

        u2 = simulate_offline(u1, args['drop_rate'])
        if len(u2) < args['threshold']:
            print("===== Agreement terminated 2=====")
            sys.exit(1)
        else:
            all_anonymous_model_upload_list = []
            all_encrypted_shared_values = []
            for client in u2:
                anonymous_model_upload_list = myClients.clients_set[client].get_anonymous_model_upload_list()
                all_anonymous_model_upload_list.append(anonymous_model_upload_list)

                encrypted_shared_values = myClients.clients_set[client].get_encrypted_shared_values()
                all_encrypted_shared_values.append(encrypted_shared_values)

            for client in u2:
                decryptable_shared_values = []
                for each_dict in all_encrypted_shared_values:
                    decryptable_shared_values.append(each_dict[client])
                myClients.clients_set[client].receive_decryptable_shared_values(decryptable_shared_values)


        u3 = simulate_offline(u2, args['drop_rate'])
        if len(u3) < args['threshold']:
            print("===== Agreement terminated 3=====")
            sys.exit(1)
        else:
            # Round 2
            summed_values_dict = {}
            for client in u3:
                summed_shared_values = myClients.clients_set[client].decrypt_and_sum_shared_values()
                summed_values_dict[client] = summed_shared_values


            # SS.Recon
            sum_of_secrets = 0
            for i_keys in u3:
                i = int(i_keys[6:])
                product_of_secrets = summed_values_dict[i_keys]
                for j_keys in u3:
                    j = int(j_keys[6:])
                    if i != j:
                        product_in_j = (-j / (i - j))
                        product_of_secrets *= product_in_j
                sum_of_secrets += product_of_secrets
            sum_of_secrets = round(sum_of_secrets)


            original_model_gradient_list = []
            # calculate the folds(g) and gradiant(W) separately, then multiply then together

            summed_model_mask = 0
            for client in u2:
                summed_model_mask += myClients.clients_set[client].model_mask

            # the right part of the denominator (constant
            temp_denominator_right = copy.deepcopy(global_parameters)
            for times in range(len(u2) - 2): # besides the first time above
                for key, var in global_parameters.items():
                    temp_denominator_right[key] *= var.clone()

            for item_count in range(1, args['k_positions'] * Np + 1):

                # the left part of the denominator
                temp_exponent = sum_of_secrets + item_count * len(u2)
                temp_denominator_left = Power(param['g'], temp_exponent)

                '''Below is the calculation of parts of aggregation_model_list '''
                fold_part = Power(param['g'], 0)
                gradiant_part = {}
                for each_model_upload_list in all_anonymous_model_upload_list:
                    #fold_part
                    fold_part *= each_model_upload_list[item_count - 1][0]

                    # gradiant_part
                    if gradiant_part == {}: # first iteration
                        gradiant_part = copy.deepcopy(each_model_upload_list[item_count - 1][1])
                    else: # not the first time
                        for key, var in each_model_upload_list[item_count - 1][1].items():
                            gradiant_part[key] *= var.clone()
                '''Above is the calculation of parts of aggregation_model_list '''

                # each item in original_model_gradient_list is a model parameters with layers
                temp_dict = {}
                for key in temp_denominator_right.keys():
                    temp_result_for_fold = (fold_part / temp_denominator_left).get_result()
                    temp_result_for_gradient = gradiant_part[key] / temp_denominator_right[key]
                    overall_results = temp_result_for_fold * temp_result_for_gradient - global_parameters[key]
                    temp_dict[key] = torch.nan_to_num(overall_results) # remove nan

                proportion_threshold = 0.85
                count_of_greater_than_threshold = []
                target_count = 0
                total_elements = 0
                for key, var in temp_dict.items():
                    # target_count += torch.sum((var <= 1e-10) & (var >= -1e-10))
                    target_count += torch.sum(var == 0)
                    total_elements += var.numel()
                proportion = target_count / total_elements
                count_of_greater_than_threshold.append(proportion >= proportion_threshold)

                if torch.tensor(False) in count_of_greater_than_threshold:
                    original_model_gradient_list.append(temp_dict)
                else:
                    original_model_gradient_list.append(0)

            if 0 in original_model_gradient_list: # to make sure 0 is in the list, to avoid error
                if len(original_model_gradient_list) - original_model_gradient_list.count(0) == len(u2):

                    original_model_gradient_list = [x for x in original_model_gradient_list if x != 0] # remove all 0
                    future_global_parameters = copy.deepcopy(original_model_gradient_list[0])
                    for each_gradiant in original_model_gradient_list[1:]:
                        for key, var in each_gradiant.items():
                            future_global_parameters[key] += var
                    for key in future_global_parameters.keys():
                        global_parameters[key] += future_global_parameters[key] / len(u2)
                else:
                    print("===== Agreement terminated 4 =====")
                    sys.exit(1)

        with torch.no_grad():
            if (comm_round + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print(' ==> Accuracy: {:.2%}'.format(sum_accu / num))


    if 0:
       if (comm_round + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                comm_round, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_participants'],
                                                                                                args['cfraction'])))


