import os
import sys
import argparse
from tqdm import tqdm
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
parser.add_argument('-np', '--num_of_participants', type=int, default=30, help='numer of the clients')
parser.add_argument('-kp', '--k_positions', type=int, default=2, help='number of positions that each participant can choose')

parser.add_argument('-cf', '--cfraction', type=float, default=0.9, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-dr', '--drop_rate', type=float, default=0.1, help='drop rate')
parser.add_argument('-t', '--threshold', type=float, default=15, help='the minimum number of hosts that can complete the iteration')

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

    a = random.randint(1, p)

    g = random.randint(2, 10)  # generator

    if binary_operator == '+':
        G = list(set(range(0, 300, g)))
    if binary_operator == '*':
        G = [g**i for i in range(20)]

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
    print("===== Params generation completed =====")

    Np = int(max(args['num_of_participants'] * args['cfraction'], 1))  # number in communication
    data_positions = list(range(1, args['k_positions'] * Np + 1))
    random.shuffle(data_positions)

    private_key = random.randint(1, param['p'])
    if param["b"] == "+":
        public_key = param['g'] * private_key
    elif param["b"] == "*":
        public_key = param['g'] ** private_key

    from clients import ClientsGroup, Clients, bilinear_pairing_function
    Clients.param = param
    Clients.k_positions = args['k_positions']

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_participants'], dev)
    testDataLoader = myClients.test_data_loader
    clients_set = myClients.get_clients()
    Clients.clients_set = clients_set
    print("===== Clients generation completed =====\n")


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


    for comm_round in range(args['num_comm']):
        print("== Communicate round {} ==".format(comm_round + 1))

        order = np.random.permutation(args['num_of_participants']) # Shuffle the clients
        clients_in_comm = ['client{}'.format(comm_round) for comm_round in order[0:Np]]
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
            temp_exponent = param['a'] ** (k_plus_Np - len(myClients.clients_set[each_client].request_parameters))

            left_side = bilinear_pairing_function(token, verification_information)
            if param["b"] == "+":
                right_side = bilinear_pairing_function(param['g'], param['h'] * temp_exponent)
            elif param["b"] == "*":
                right_side = bilinear_pairing_function(param['g'], param['h'] ** temp_exponent)

            if left_side != right_side: # 双线性配对函数 bilinear pairing function
                print("===== Agreement terminated 1=====")
                sys.exit(1)
            else:
                random_mask = random.randint(1, param['p'])
                # OT.Enc
                secret_list = []
                for count in range(args['k_positions'] * Np): # count == n
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
                                                                         random_mask * param['h']) * data_positions[count])
                        elif param["b"] == "*":
                            secret_list.append(bilinear_pairing_function(param['g'] ** (1 / (param['a'] + count)),
                                                                         random_mask * param['h']) * data_positions[count])
                    count += 1

                myClients.clients_set[each_client].set_secret_list(secret_list)
                myClients.clients_set[each_client].decrypt_secret()

        '''=====数据匿名收集阶段====='''
        # Round 1
        u1 = clients_in_comm
        #random.shuffle(u1)
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


            aggregation_model_list = [] # Lw

            # part 2
            part_2 = {}
            for layer, model_parameter in global_parameters.items():
                new_model_parameter = model_parameter.clone() ** (len(u2) - 1)
                part_2[layer] = new_model_parameter

            for item_count in range(1, args['k_positions'] * Np + 1):

                # part 1
                temp_sum = 0
                for client in u2:
                    temp_sum += myClients.clients_set[client].model_mask + len(u2) * item_count
                if param["b"] == "+":
                    part_1 = param['g'] * temp_sum
                elif param["b"] == "*":
                    part_1 = param['g'] ** temp_sum

                # part 3
                part_3 = 0
                for client in clients_in_comm:
                    if item_count in myClients.clients_set[client].request_parameters:
                        part_3 = myClients.clients_set[client].local_parameters
                        break

                '''try:
                    index = data_positions.index(item_count) # index is j
                except ValueError:
                    print("Couldn't find the index of the element")
                part_3 = myClients.clients_set["client{}".format(index)].local_parameters'''

                # Multiply the three above
                temp_dict = {}
                if part_3 == 0:
                    aggregation_model_list.append(0)
                else:
                    for layer in part_2.keys():
                        product = part_1 * part_2[layer].clone() * part_3[layer].clone()
                        temp_dict[layer] = product.clone()

                    aggregation_model_list.append(temp_dict)


            # SS.Recon
            sum_of_secrets = 1 # 乘积
            for i in u3: # 但是之前累加的分享值是u2的主机
                temp_sum = 0
                for j in u3[:args['threshold']]: # 选择分享值是从j属于u3中选取，所以用u3截
                    if ((int(i[6:]) + 1) - (int(j[6:]) + 1)) != 0:
                        temp_sum += (-(int(j[6:]) + 1) / ((int(i[6:]) + 1) - (int(j[6:]) + 1))) * summed_values_dict[i]
                sum_of_secrets *= temp_sum


            original_model_gradient_list = []

            # the right part of the denominator (constant
            temp_denominator_right = part_2
            for item_count in range(1, args['k_positions'] * Np + 1):

                # the left part of the denominator
                temp_sum = 0
                for client in u2:
                    temp_sum += myClients.clients_set[client].model_mask + len(u2) * item_count
                if param["b"] == "+":
                    temp_denominator_left = param['g'] * temp_sum
                elif param["b"] == "*":
                    temp_denominator_left = param['g'] ** temp_sum

                # each item in original_model_gradient_list is a model parameters with layers
                if aggregation_model_list[item_count - 1] == 0:
                    original_model_gradient_list.append(0)
                else:
                    temp_dict = {}
                    for var in temp_denominator_right.keys():
                        temp_dict[var] = aggregation_model_list[item_count - 1][var].clone() \
                                           / (temp_denominator_left * temp_denominator_right[var].clone()) - global_parameters[var].clone()

                    original_model_gradient_list.append(temp_dict)

            print("0：{}".format(original_model_gradient_list.count(0)))
            print("非空：{}".format(len(original_model_gradient_list) - original_model_gradient_list.count(0)))
            print("u2：{}".format(len(u2)))
            if 0 in original_model_gradient_list: # to make sure 0 is in the list, to avoid error
                if len(original_model_gradient_list) - original_model_gradient_list.count(0) == len(u2):
                    pass
                else:
                    pass
                    #print("===== Agreement terminated 4 =====")
                    #sys.exit(1)

            for var in temp_denominator_right.keys():
                for each_parameter in original_model_gradient_list:
                    if each_parameter != 0:
                        global_parameters[var] += each_parameter[var]
                number_of_zero_in_original_model_gradient_list = len(original_model_gradient_list)\
                                                                 - original_model_gradient_list.count(0)
                global_parameters[var] = global_parameters[var] / number_of_zero_in_original_model_gradient_list


        '''sum_parameters = None
        for client in tqdm(clients_in_comm):

            local_parameters = myClients.clients_set[client].local_update(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            if sum_parameters is None: # First iteration
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else: # Not first iteration
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / Np)'''

        with torch.no_grad():
            if 1:
            #if (comm_round + 1) % args['val_freq'] == 0:
                net.load_state_dict(local_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))


    if 0:
       if (comm_round + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                comm_round, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_participants'],
                                                                                                args['cfraction'])))


