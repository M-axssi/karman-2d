import math
import os, argparse, pickle, glob
import phi.field
import torch.distributed
from phi.physics._boundaries import Domain, OPEN
from phi.torch.flow import *
from phi import math
import torch.nn.functional as F
import torch.nn as nn


def get_lr(epoch, cur_lr):
    lr = cur_lr
    if epoch == 181:
        lr *= 0.5
    elif epoch == 161:
        lr *= 1e-1
    elif epoch == 121:
        lr *= 1e-1
    elif epoch == 81:
        lr *= 1e-1
    return lr


class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=1, padding=2),
            nn.LeakyReLU()
        )
        self.block0 = nn.Sequential(
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2)
        )
        self.LR0 = nn.LeakyReLU()
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2)
        )
        self.LR1 = nn.LeakyReLU()
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2)
        )
        self.LR2 = nn.LeakyReLU()
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2)
        )
        self.LR3 = nn.LeakyReLU()
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2)
        )
        self.LR4 = nn.LeakyReLU()
        self.out = nn.Conv2d(32, 2, (5, 5), stride=1, padding=2)

    def forward(self, x):
        out1 = self.pre(x)
        out2 = self.block0(out1) + out1
        out3 = self.LR0(out2)
        out4 = self.block1(out3) + out3
        out5 = self.LR1(out4)
        out6 = self.block2(out5) + out5
        out7 = self.LR2(out6)
        out8 = self.block3(out7) + out7
        out9 = self.LR3(out8)
        out10 = self.block4(out9) + out9
        out11 = self.LR4(out10)
        return self.out(out11)


class PRE_Dataset:
    def __init__(self, domain, num_sim, dirpath, begin_step, num_step):
        self.sim_path = sorted(glob.glob(dirpath + "/sim_0*"))[0:num_sim]
        self.corr_velo_path = [sorted(glob.glob(asim + "/corr_velo_0*.npz"))[begin_step - 1:begin_step + num_step] for
                               asim in self.sim_path]
        self.source_velo_path = [sorted(glob.glob(asim + "/source_velo_0*.npz"))[begin_step - 1:begin_step + num_step]
                                 for asim in self.sim_path]
        self.domain = domain
        self.num_sim = num_sim
        self.num_step = num_step
        self.Res = {}

        for asim in self.sim_path:
            with open(asim + "/params.pickle", 'rb')as f:
                params = pickle.load(f)
                self.Res[asim] = params["re"]

        self.Data = {
            asim: [
                (
                    phi.field.read(self.source_velo_path[j][i]).vector['y'].values.numpy(
                        ('y', 'x')),
                    phi.field.read(self.source_velo_path[j][i]).vector['x'].values.numpy(
                        ('y', 'x')),
                    phi.field.read(self.corr_velo_path[j][i]).vector['y'].values.numpy(
                        ('y', 'x')),
                    phi.field.read(self.corr_velo_path[j][i]).vector['x'].values.numpy(
                        ('y', 'x'))
                )
                for i in range(num_step)]
            for j, asim in enumerate(self.sim_path)
        }

        self.Data_std = [
            np.std(np.concatenate([np.absolute(self.Data[asim][i][0].reshape(-1)) for asim in self.sim_path for i in
                                   range(self.num_step)], axis=-1)),
            np.std(np.concatenate([np.absolute(self.Data[asim][i][1].reshape(-1)) for asim in self.sim_path for i in
                                   range(self.num_step)], axis=-1)),
            np.std(np.absolute([self.Res[asim] for asim in self.sim_path]))
        ]

    def Get_Data(self, sim, begin, steps=1):
        Re = math.stack(
            [
                domain.scalar_grid(self.Res[self.sim_path[sim]]).values for _ in
                range(params['batch_size'])
            ], math.channel('batch')
        )
        source_velocity_y = [self.Data[self.sim_path[sim]][begin + k][0] for k in range(steps)]
        source_velocity_x = [self.Data[self.sim_path[sim]][begin + k][1] for k in range(steps)]
        corr_velocity_y = [self.Data[self.sim_path[sim]][begin + k][2] for k in range(steps)]
        corr_velocity_x = [self.Data[self.sim_path[sim]][begin + k][3] for k in range(steps)]
        return Re, source_velocity_y, source_velocity_x, corr_velocity_y, corr_velocity_x


def to_feature(V, Re):
    return math.stack(
        [
            V.vector['x'].x[:-1].values,
            V.vector['y'].y[:-1].values,
            Re
        ], math.channel('channel')
    )


def to_staggered(output, domain):
    return domain.staggered_grid(
        math.stack(
            [
                math.tensor(F.pad(output[:, 1], (0, 0, 0, 1, 0, 0), mode='constant', value=0), math.batch('batch'),
                            spatial('y,x')),
                math.tensor(F.pad(output[:, 0], (0, 1, 0, 0, 0, 0), mode='constant', value=0), math.batch('batch'),
                            spatial('y,x'))
            ]
            , math.channel('vector')
        )
    )


def train_batch(t_Res, t_SV, t_CV, multi_step):
    optimizer.zero_grad()
    input = to_feature(t_SV, t_Res)
    input /= math.tensor([MyDataset.Data_std[1], MyDataset.Data_std[0], MyDataset.Data_std[2]],
                         channel('channel'))
    output = Net.forward(input.native(['batch', 'channel', 'y', 'x']))
    output[:, 0] *= MyDataset.Data_std[1]
    output[:, 1] *= MyDataset.Data_std[0]
    corr_value = to_staggered(output, domain)

    loss = torch.nn.MSELoss(reduction='sum').to(TORCH.get_default_device().ref)
    loss_velocity_x = loss(
        corr_value.vector['x'].values.native(('batch', 'y', 'x')),
        t_CV.vector['x'].values.native(('batch', 'y', 'x'))
    ) / MyDataset.Data_std[1]
    loss_velocity_y = loss(
        corr_value.vector['y'].values.native(('batch', 'y', 'x')),
        t_CV.vector['y'].values.native(('batch', 'y', 'x'))
    ) / MyDataset.Data_std[0]

    total_loss = (loss_velocity_x + loss_velocity_y) / multi_step
    total_loss.backward()
    optimizer.step()
    return total_loss


params = {}
parser = argparse.ArgumentParser(description="Parameter parser.")
parser.add_argument('-r', '--res', type=int, default=32, help="Resolution of x axis")
parser.add_argument('-l', '--len', type=int, default=100, help="Length of x axis")
parser.add_argument('--steps', type=int, default=500, help="Total steps")
parser.add_argument('--initial_step', type=int, default=1000, help="Initial step")
parser.add_argument('-s', '--sim', type=int, default=6, help="Number of simulation")
parser.add_argument('-e', '--epoch', type=int, default=500, help="Number of epoch")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_path', type=str, default=None, help="Path of model")
parser.add_argument('--device', type=str, default='CPU', help="Device for training")
parser.add_argument('--input', type=str, default='./Pre-Data', help="Path for input data")
parser.add_argument('--output', type=str, default='./model/PRE.pth', help="Path for output model")
args = parser.parse_args()
params.update(vars(args))

TORCH.set_default_device(params['device'])

domain = Domain(y=params['res'] * 2, x=params['res'], bounds=Box[0:params['len'] * 2, 0:params['len']], boundaries=OPEN)
checkpoint = None
Net = Mynet()
if params['model_path'] is not None:
    checkpoint = torch.load(params['model_path'], map_location=TORCH.get_default_device().ref)
    Net.load_state_dict(checkpoint['model'])
Net.to(TORCH.get_default_device().ref)
cur_lr = 0.0001
optimizer = optim.Adam(Net.parameters(), lr=cur_lr)
if params['model_path'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])

MyDataset = PRE_Dataset(domain, params['sim'], params['input'], params['initial_step'], params['steps'])
print("Finish read data")

if not os.path.exists(os.path.dirname(params['output'])):
    os.mkdir(os.path.dirname(params['output']))

for i in range(params['epoch']):
    temp = get_lr(i, cur_lr)
    if temp != cur_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = temp
    cur_lr = temp
    for j in range(params['sim']):
        for k in range(params['steps'] // params['batch_size']):
            Res, SVy, SVx, CVy, CVx = MyDataset.Get_Data(j, k * params['batch_size'], params['batch_size'])
            t_Res = Res
            t_SV = domain.staggered_grid(
                math.stack(
                    [
                        math.tensor(SVy, batch('batch'), spatial('y,x')),
                        math.tensor(SVx, batch('batch'), spatial('y,x'))
                    ], channel('vector')
                )
            )
            t_CV = domain.staggered_grid(
                math.stack(
                    [
                        math.tensor(CVy, batch('batch'), spatial('y,x')),
                        math.tensor(CVx, batch('batch'), spatial('y,x'))
                    ], channel('vector')
                )
            )

            print(f"The mean loss of epoch{i},sim{j},batch{k} is "
                  f"{train_batch(t_Res, t_SV, t_CV, params['batch_size'])}")
    if i % 50 == 49:
        torch.save({'model': Net.state_dict(), 'optimizer': optimizer.state_dict()}, params['output'])

torch.save({'model': Net.state_dict(), 'optimizer': optimizer.state_dict()}, params['output'])
