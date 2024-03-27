import numpy as np
import torch
import random
from tqdm import tqdm
from decimal import getcontext, Decimal
from sympy import symbols, Eq, solve
from sympy import isprime, nextprime
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
from functools import reduce

def correct_inaccurate_round(float):
    threshold = 1e-6
    round_float = round(float)
    if abs(float - round_float) > threshold:
        return float
    else:
        return round_float


def bilinear_pairing_function(a, b):
    #a, b = Decimal(a), Decimal(b)
    if Clients.param["b"] == "+":
        result = (a * b)
        #result = Clients.param['g'] * (a * b)
        #result = pow(Clients.param['g'], a * b, Clients.param['p'])

    elif Clients.param["b"] == "*":
        result = Clients.param['g'] ** (a * b)
        #result = pow(a, b, Clients.param['p'])

    return correct_inaccurate_round(result)


def elgamal_encrypt(secret, pubilc_key):
    k = nextprime(random.randint(1, 5)) # 1 < k < p-1
    pubilc_key = pubilc_key % Clients.param['p']

    if Clients.param["b"] == "+":
        c1 = (Clients.param['g'] * k) % Clients.param['p']
        c2 = (pubilc_key * k + secret) % Clients.param['p']
    elif Clients.param["b"] == "*":
        c1 = (Clients.param['g'] ** k) % Clients.param['p']
        c2 = (pubilc_key ** k * secret) % Clients.param['p']

    return [c1, c2]


def elgamal_decrypt(messages, private_key):
    c1, c2 = messages[0], messages[1]

    if Clients.param["b"] == "+":
        s = (c1 * private_key) % Clients.param['p']
        return c2 - s % Clients.param['p']
    elif Clients.param["b"] == "*":
        s = c1 ** (Clients.param['p'] - 2) % Clients.param['p']
        return c2 * s ** private_key % Clients.param['p']




class Clients(object):
    param = {}
    k_positions = None
    clients_in_comm = []
    clients_set = {}

    def __init__(self, trainDataSet, public_parameter, dev, client_private_key):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

        '''=====Stage 1====='''
        self.public_parameter = public_parameter # uj, not g ** uj
        self.client_private_key = client_private_key # ski
        if Clients.param["b"] == "+":
            self.client_public_key = Clients.param['g'] * client_private_key
        elif Clients.param["b"] == "*":
            self.client_public_key = Clients.param['g'] ** client_private_key

        '''=====Stage 2====='''
        self.request_parameters = []
        self.whether_picked_in_round1 = False

        self.secret_list = []
        self.position_list = []

        '''=====Stage 3====='''
        # Round 1
        self.local_parameters = {}
        self.model_mask = 0
        self.anonymous_model_upload_list = []
        self.encrypted_shared_values = {}
        self.decryptable_shared_values = []
        # Round 2



    def local_update(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()


    def take_out_from_request_collection(self, request_collection):
        amount_to_take_out = random.randint(1, Clients.k_positions)
        if len(request_collection) <= amount_to_take_out:
            return [], request_collection
        else:
            elements_to_remove = random.sample(request_collection, amount_to_take_out)
            new_request_collection = [item for item in request_collection if item not in elements_to_remove]

            return elements_to_remove, new_request_collection


    def round1_first_client(self):
        self.whether_picked_in_round1 = True

        # i
        request_collection = list(range(1, Clients.k_positions * len(Clients.clients_in_comm) + 1))

        # ii
        self.request_parameters = []
        request_parameters, request_collection = self.take_out_from_request_collection(request_collection)
        self.request_parameters = request_parameters

        b = random.randint(1, 50) # b belongs to set Z
        total_public_parameters = 0
        for each_client in Clients.clients_in_comm:
            total_public_parameters += Clients.clients_set[each_client].public_parameter

        if Clients.param["b"] == "+":
            timestep = Clients.param['g'] * (b * (total_public_parameters - self.public_parameter))
        elif Clients.param["b"] == "*":
            timestep = Clients.param['g'] ** (b * (total_public_parameters - self.public_parameter))


        # iii
        next_client_public_key = 0
        for each_client in Clients.clients_in_comm:
            if Clients.clients_set[each_client].whether_picked_in_round1 == False:
                next_client_public_key = Clients.clients_set[each_client].client_public_key
                break

        encrypted_request_collection = []
        for element in request_collection:
            encrypted_element = elgamal_encrypt(element, next_client_public_key)
            encrypted_request_collection.append(encrypted_element)

        if Clients.param["b"] == "+":
            return each_client, [encrypted_request_collection, timestep, Clients.param['g'] * b]
        elif Clients.param["b"] == "*":
            return each_client, [encrypted_request_collection, timestep, Clients.param['g'] ** b]


    def round1_other_clients(self, encrypted_request_collection, timestep, gb):
        self.whether_picked_in_round1 = True

        # i (no need to re-select
        pass

        # ii
        request_collection = []
        for encrypted_element in encrypted_request_collection:
            decrypted_element = elgamal_decrypt(encrypted_element, self.client_private_key)
            request_collection.append(decrypted_element)

        request_parameters, request_collection = self.take_out_from_request_collection(request_collection)
        self.request_parameters = request_parameters
        # iii
        if Clients.param["b"] == "+":
            timestep = timestep - gb * self.public_parameter
        elif Clients.param["b"] == "*":
            timestep = timestep / gb ** self.public_parameter

        # iv
        next_client_public_key = 0
        if timestep == (0 if Clients.param["b"] == "+" else 1 if Clients.param["b"] == "*" else ""):
            return 0, 0
        else:
            for each_client in Clients.clients_in_comm:
                if not Clients.clients_set[each_client].whether_picked_in_round1:
                    next_client_public_key = Clients.clients_set[each_client].client_public_key
                    break


            encrypted_request_collection = []
            for element in request_collection:
                encrypted_element = elgamal_encrypt(element, next_client_public_key)
                encrypted_request_collection.append(encrypted_element)

            return each_client, [encrypted_request_collection, timestep, gb]


    def get_token_and_verification_information(self):
        temp_sum_list = [element+Clients.param['a'] for element in self.request_parameters]
        temp_product = reduce(lambda x, y: x*y, temp_sum_list) # multiply the items in the list
        k_plus_Np = Clients.k_positions * len(Clients.clients_in_comm)
        temp_exponent = Clients.param['a'] * (k_plus_Np - len(self.request_parameters))

        # OT.Token
        if Clients.param["b"] == "+":
            token = Clients.param['g'] * (self.client_private_key / temp_product)  # Toki
            verification_information = Clients.param['h'] * ((temp_product * temp_exponent) / self.client_private_key)  # hi
        elif Clients.param["b"] == "*":
            token = Clients.param['g'] ** (self.client_private_key / temp_product)  # Toki
            verification_information = Clients.param['h'] ** ((temp_product * temp_exponent) / self.client_private_key)  # hi


        return token, verification_information, len(self.request_parameters)


    def set_secret_list(self, secret_list):
        self.secret_list = secret_list


    def decrypt_secret(self):
        # OT.Dec
        self.position_list = []
        a_plus_ld = [element + Clients.param['a'] for element in self.request_parameters]
        temp_product = reduce(lambda x, y: x*y, a_plus_ld)

        for count in range(1, Clients.k_positions * len(Clients.clients_in_comm)):
            Cn = self.secret_list[count]

            if Clients.param["b"] == "+":
                right_side = Clients.param['h'] * (temp_product / (Clients.param['a'] + count))
                sn = Cn / bilinear_pairing_function(self.secret_list[0], right_side * (1 / self.client_private_key))
            elif Clients.param["b"] == "*":
                right_side = Clients.param['h'] ** (temp_product / (Clients.param['a'] + count))
                sn = Cn / bilinear_pairing_function(self.secret_list[0], right_side ** (1 / self.client_private_key))

            self.position_list.append(correct_inaccurate_round(sn))


    def generate_anonymous_model_upload_list(self, global_parameters, local_parameters):
        def multiply_tensor(dict, multiplier):
            new_dict = {}
            for key in dict:
                new_dict[key] = dict[key].clone() * multiplier
            return new_dict
        self.local_parameters = local_parameters
        self.model_mask = random.randint(1, int(str(Clients.param['p'])[:4]))
        #self.model_mask = random.randint(1, Clients.param['p'])
        random_position = random.choice(self.position_list)

        self.anonymous_model_upload_list = []
        for count in range(1, Clients.k_positions * len(Clients.clients_in_comm) + 1):
            if count == random_position:
                if Clients.param["b"] == "+":
                    item = multiply_tensor(local_parameters, Clients.param['g'] * (self.model_mask + count))
                elif Clients.param["b"] == "*":
                    item = multiply_tensor(local_parameters, Clients.param['g'] ** (self.model_mask + count))
            else:
                if Clients.param["b"] == "+":
                    item = multiply_tensor(global_parameters, Clients.param['g'] * (self.model_mask + count))
                elif Clients.param["b"] == "*":
                    item = multiply_tensor(global_parameters, Clients.param['g'] ** (self.model_mask + count))
            self.anonymous_model_upload_list.append(item)


    def get_anonymous_model_upload_list(self):
        return self.anonymous_model_upload_list


    def generate_and_encrypt_shared_values(self, t):
        '''generation'''
        coefficients = [random.randint(-10, 10) for _ in range(t-1)]
        def multiple_equations(highest_degree_of_function, coefficients, x):
            y = 0
            for degree in range(1, highest_degree_of_function):
                y += coefficients[degree - 1] * x ** degree
            return y

        shared_values = {}
        for each_client in Clients.clients_in_comm:
            client_number = int(each_client[6:]) + 1
            shared_values[each_client] = multiple_equations(t-1, coefficients, client_number) + self.model_mask
            # the self.model_mask is the constant s in the function

        '''encryption'''
        self.encrypted_shared_values = {} # {'client24': 365590165707817800, 'client12': 89546683720116...
        for each_client, shared_values in shared_values.items():
            public_key_of_pj = Clients.clients_set[each_client].client_public_key
            encrypted_value = elgamal_encrypt(shared_values, public_key_of_pj)
            self.encrypted_shared_values[each_client] = encrypted_value


    def get_encrypted_shared_values(self):
        return self.encrypted_shared_values


    def receive_decryptable_shared_values(self, decryptable_shared_values):
        self.decryptable_shared_values = decryptable_shared_values


    def decrypt_and_sum_shared_values(self):
        decrypt_values = []
        for cipher in self.decryptable_shared_values:
            value = elgamal_decrypt(cipher, self.client_private_key)
            decrypt_values.append(value)

        summed_shared_values = sum(decrypt_values)
        return summed_shared_values















class ClientsGroup(object):

    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.allocate_data_set_balance()


    def allocate_data_set_balance(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)
            someone = Clients(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)),
                              random.randint(1, 3), self.dev, random.randint(1, int(str(Clients.param['p'])[:8])))
            self.clients_set['client{}'.format(i)] = someone

    def get_clients(self):
        return self.clients_set

    def round1(self, Pi):
        for each_client in Clients.clients_in_comm:
            self.clients_set[each_client].whether_picked_in_round1 = False

        next_client, returns = self.clients_set[Pi].round1_first_client()

        while True:
            next_client, returns = self.clients_set[next_client].round1_other_clients(*returns)
            if next_client == 0:  # round 1 all done! Time for position selecting
                break

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


