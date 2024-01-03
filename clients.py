import numpy as np
import torch
import random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet


class Clients(object):
    k_positions = None
    clients_in_comm = []
    clients_set = {}

    def __init__(self, trainDataSet, public_parameter, dev, xi):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.public_parameter = public_parameter # uj, not g ** uj
        self.ski = xi
        self.pki = ClientsGroup.param['g'] ** xi

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
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


    def takeOutFromRequestCollection(self, request_collection):
            elements_to_remove = random.sample(request_collection, random.randint(1, Clients.k_positions))
            for element in elements_to_remove:
                request_collection.remove(element)
            return elements_to_remove, request_collection

    def round1_firstClient(self):
        request_collection = list(range(1, k_positions * Clients.clients_in_comm + 1))

        request_parameters, request_collection = self.takeOutFromRequestCollection(request_collection)
        b = random.randint(1, 50) # b belongs to set Z
        total_public_parameters = 0
        for each_client in Clients.clients_in_comm:
            total_public_parameters += Clients.clients_set[each_client].public_parameter

        timestep = param['g'] ** (b * (total_public_parameters - self.public_parameter))


    def round1_otherClients(self):

    def round1(self):
        self.round1_firstClient()


    def local_val(self):
        pass








class ClientsGenerator(object):

    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.dataSetBalanceAllocation()


    def dataSetBalanceAllocation(self):
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
            someone = Clients(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), random.randint(0, 100), self.dev, random.choice(self.Zp), self.param)
            self.clients_set['client{}'.format(i)] = someone

    def getClients(self):
        return self.clients_set

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


