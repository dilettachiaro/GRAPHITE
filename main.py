import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
from utils.utils import CustomDataset, align_ts_kg, create_dataset, metric_test
from utils.models import GRU_model, LSTM_model, GraphConvolution, MLP


# define model of generator and discriminator
class Generator(nn.Module):
    def __init__(self, step, is_cuda=True):
        super(Generator, self).__init__()
        self.step = step
        self.is_cuda = is_cuda
        self.GRU_model_1 = GRU_model(self.step, self.is_cuda)
        self.GRU_model_2 = GRU_model(self.step, self.is_cuda)
        self.LSTM_model = LSTM_model(self.step, self.is_cuda)

        self.GCN_model = GraphConvolution(self.step, self.is_cuda)
        self.MLP_model = MLP()

    def forward(self, x_ts, x_f, kg_1, kg_2):
        out_1 = self.GRU_model_1(x_ts[:, :, 0].unsqueeze(2)).unsqueeze(2)
        out_2 = self.GRU_model_2(x_ts[:, :, 1].unsqueeze(2)).unsqueeze(2)
        out_3 = self.LSTM_model(x_ts[:, :, 2].unsqueeze(2)).unsqueeze(2)

        out_gru = torch.cat((out_1, out_2, out_3), dim=2)

        out_gcn = self.GCN_model(x_f, kg_1, kg_2)

        out_cat = torch.cat((out_gru, out_gcn), dim=2)

        out = self.MLP_model(out_cat)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.linear1 = nn.Linear(64, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.LeakyReLU(x)
        x = x.permute(0, 2, 1)
        x = self.linear1(x[:, -1, :])
        x = self.sigmoid(x)

        return x
    

# define gradient penalty function
def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = Variable(alpha * x.data + (1 - alpha) * y.data, requires_grad=True).cuda()
    # gradient penalty
    o = f(z)
    g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].cuda()
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
    return gp


if __name__ == '__main__':

    # set random seed
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # define hyper-parameters
    window=240      # history window size, 10 days
    step = 24       # prediction horizon, 1 day
    batch_size = 64
    epochs = 100
    lr = 0.001

    # get data path
    train_ts_path = 'GAN-GNN/data_processing/train_ts_dublin.pkl'
    test_ts_path = 'GAN-GNN/data_processing/test_ts_dublin.pkl'
    kg_path = 'GAN-GNN/data_processing/knowledge_graphs_dublin.pkl'


    # align time series data and knowledge graph data
    train_data, test_data = align_ts_kg(train_ts_path, test_ts_path, kg_path)

    # create dataset
    train_X, train_Y, test_X, test_Y = create_dataset(train_data, test_data, window=window, step=step)

    # Create custom dataset for dataloader
    train_dataset = CustomDataset(train_X, train_Y)
    test_dataset = CustomDataset(test_X, test_Y)

    traindata_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testdata_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # define model
    generator = Generator(step=step, is_cuda=True).cuda()
    discriminator = Discriminator().cuda()


    # define loss function
    loss_func_G = nn.MSELoss()
    loss_func_D = nn.BCELoss()


    # define optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)


    # train
    generator_loss = []
    discriminator_loss = []
    pred_list = []
    real_list = []

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for step, (x_ts, x_f, kg_1, kg_2, y) in enumerate(traindata_loader):

            x_ts, x_f, kg_1, kg_2 = Variable(x_ts), Variable(x_f), Variable(kg_1), Variable(kg_2)
            target = Variable(y)

            x_ts = x_ts.cuda()
            x_f = x_f.cuda()
            kg_1 = kg_1.cuda()
            kg_2 = kg_2.cuda()
            target = target.cuda()

            pred = generator(x_ts, x_f, kg_1, kg_2)

            fake_data = Variable(torch.cat((x_ts, pred), dim=1)).cuda()
            real_data = Variable(torch.cat((x_ts, target), dim=1)).cuda()

            real_label = Variable(torch.zeros((real_data.shape[0], 1))).cuda()
            fake_label = Variable(torch.ones((fake_data.shape[0], 1))).cuda()


            optimizer_D.zero_grad()
            # train with real data
            output_real = discriminator(real_data)
            loss_real = loss_func_D(output_real, real_label)

            # train with fake data
            output_fake = discriminator(fake_data)
            loss_fake = loss_func_D(output_fake, fake_label)

            loss_D = (loss_real + loss_fake)/2

            # gradient penalty
            gp = gradient_penalty(real_data, fake_data, discriminator)
            loss_D = loss_D + 10 * gp

            loss_D.backward()
            optimizer_D.step()


            for _ in range(5):
                # train generator
                optimizer_G.zero_grad()
                pred = generator(x_ts, x_f, kg_1, kg_2)

                mse_loss = loss_func_G(pred, target)

                fake_data = Variable(torch.cat((x_ts, pred), dim=1)).cuda()
                output_fake = discriminator(fake_data)
                binary_loss = loss_func_D(output_fake, real_label)

                loss_G = mse_loss + binary_loss
                
                loss_G.backward()
                optimizer_G.step()

        #save losses
        generator_loss.append(loss_G.item())
        discriminator_loss.append((loss_D).item())

        print('Epoch: ', epoch, '| generator loss: ', loss_G.item(), '| discriminator loss: ', (loss_D).item())


    #plot losses
    import matplotlib.pyplot as plt
    plt.plot(generator_loss, label='generator loss')
    plt.plot(discriminator_loss, label='discriminator loss')
    plt.legend()
    plt.show()


    # test
    generator.eval()
    discriminator.eval()

    pred_test_list = []
    real_test_list = []


    with torch.no_grad():
        for step, (x_ts, x_f, kg_1, kg_2, y) in enumerate(testdata_loader):

            x_ts, x_f, kg_1, kg_2 = Variable(x_ts), Variable(x_f), Variable(kg_1), Variable(kg_2)
            # y should be decreased 1 dimension
            target = Variable(y)

            x_ts = x_ts.cuda()
            x_f = x_f.cuda()
            kg_1 = kg_1.cuda()
            kg_2 = kg_2.cuda()
            target = target.cuda()

            pred = generator(x_ts, x_f, kg_1, kg_2)

            fake_data = Variable(torch.cat((x_ts, pred), dim=1)).cuda()
            real_data = Variable(torch.cat((x_ts, target), dim=1)).cuda()

            pred_test_list.append(pred.cpu().detach().numpy())
            real_test_list.append(target.cpu().detach().numpy())


    # calculate metrics
    PRED_list = []
    REAL_list = []

    for i in range(len(pred_test_list)):
        PRED_list.append(pred_test_list[i])
        REAL_list.append(real_test_list[i])

    PRED = np.concatenate(PRED_list, axis=0)
    REAL = np.concatenate(REAL_list, axis=0)

    total_mse, total_rmse, total_mae, total_r2 = metric_test(PRED, REAL)

    print('mse: ', total_mse, '| rmse: ', total_rmse, '| mae: ', total_mae, '| r2: ', total_r2)

