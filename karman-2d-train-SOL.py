import math
import os, argparse, pickle, glob
import phi.field
from phi.physics._boundaries import Domain, OPEN
from phi.torch.flow import *
import torch.nn.functional as F
import torch.nn as nn


def get_lr(epoch, cur_lr):
    lr = cur_lr
    if epoch == 36:
        lr *= 0.5
    elif epoch == 31:
        lr *= 1e-1
    elif epoch == 21:
        lr *= 1e-1
    elif epoch == 11:
        lr *= 1e-1
    return lr


class KarmanFlow():
    def __init__(self, domain):
        self.domain = domain

        shape_v = self.domain.staggered_grid(0).vector['y'].shape
        vel_yBc = np.zeros(shape_v.sizes)
        vel_yBc[0:2, 0:vel_yBc.shape[1] - 1] = 1.0
        vel_yBc[0:vel_yBc.shape[0], 0:1] = 1.0
        vel_yBc[0:vel_yBc.shape[0], -1:] = 1.0
        self.vel_yBc = math.tensor(vel_yBc, shape_v)
        self.vel_yBcMask = math.tensor(np.copy(vel_yBc), shape_v)

        self.inflow = self.domain.scalar_grid(Box[5:10, 25:75])
        self.obstacles = [Obstacle(Sphere(center=[50, 50], radius=10))]

    def step(self, density_in, velocity_in, re, res, dt=1.0, make_input_divfree=False,
             make_output_divfree=True):  # , conserve_density=True):
        velocity = velocity_in
        density = density_in

        # apply viscosity
        velocity = phi.flow.diffuse.explicit(field=velocity, diffusivity=1.0 / re * dt * res * res, dt=dt)
        vel_x = velocity.vector['x']
        vel_y = velocity.vector['y']

        # apply velocity BCs
        vel_y = vel_y * (1.0 - self.vel_yBcMask) + self.vel_yBc
        velocity = self.domain.staggered_grid(phi.math.stack([vel_y.data, vel_x.data], channel('vector')))

        pressure = None
        if make_input_divfree:
            velocity, pressure = fluid.make_incompressible(velocity, self.obstacles)

        # --- Advection ---
        density = advect.semi_lagrangian(density + self.inflow, velocity, dt=dt)
        velocity = advected_velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)

        # --- Pressure solve ---
        if make_output_divfree:
            velocity, pressure = fluid.make_incompressible(velocity, self.obstacles)
        return [density, velocity]


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


class SOL_Dataset:
    def __init__(self, domain, num_sim, dirpath, begin_step, num_step, batch_size, need_downsample=True):
        self.sim_path = sorted(glob.glob(dirpath + "/sim_0*"))[0:num_sim]
        self.density_path = [sorted(glob.glob(asim + "/dens_0*.npz"))[begin_step - 1:begin_step + num_step] for asim in
                             self.sim_path]
        self.velocity_path = [sorted(glob.glob(asim + "/velo_0*.npz"))[begin_step - 1:begin_step + num_step] for asim in
                              self.sim_path]
        self.domain = domain
        self.need_downsample = need_downsample
        self.num_sim = num_sim
        self.num_step = num_step
        self.batch_size = batch_size
        self.batch_num = num_sim // batch_size
        self.cur_batch = 0
        self.cur_step = 0
        self.Res = {}

        if need_downsample:
            for j, asim in enumerate(self.sim_path):
                if not os.path.exists(asim + '/ds'):
                    os.mkdir(asim + '/ds')
                for i in range(num_step):
                    if not os.path.isfile(self.FileForDownsample(self.density_path[j][i])):
                        d = phi.field.read(file=self.density_path[j][i]).at(self.domain.scalar_grid())
                        phi.field.write(field=d, file=self.FileForDownsample(self.density_path[j][i]))
                        print("Write {}".format(self.FileForDownsample(self.density_path[j][i])))
                    if not os.path.isfile(self.FileForDownsample(self.velocity_path[j][i])):
                        d = phi.field.read(self.velocity_path[j][i]).at(self.domain.staggered_grid())
                        phi.field.write(field=d, file=self.FileForDownsample(self.velocity_path[j][i]))
                        print("Write {}".format(self.FileForDownsample(self.velocity_path[j][i])))

        for asim in self.sim_path:
            with open(asim + "/params.pickle", 'rb')as f:
                params = pickle.load(f)
                self.Res[asim] = params["re"]

        self.Data = {
            asim: [
                (
                    phi.field.read(self.FileForDownsample(self.density_path[j][i])).values.numpy(('y', 'x')),
                    phi.field.read(self.FileForDownsample(self.velocity_path[j][i])).vector['y'].values.numpy(
                        ('y', 'x')),
                    phi.field.read(self.FileForDownsample(self.velocity_path[j][i])).vector['x'].values.numpy(
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
            np.std(np.concatenate([np.absolute(self.Data[asim][i][2].reshape(-1)) for asim in self.sim_path for i in
                                   range(self.num_step)], axis=-1)),
            np.std(np.absolute([self.Res[asim] for asim in self.sim_path]))
        ]

    def FileForDownsample(self, file):
        return os.path.dirname(file) + '/ds/' + os.path.basename(file)

    def reboot(self):
        self.cur_batch = 0
        self.cur_step = 0

    def Get_Data(self, steps=1):
        Re = [
            self.Res[self.sim_path[self.cur_batch * self.batch_size + j]]
            for j in range(self.batch_size)
        ]
        density = [
            [
                self.Data[self.sim_path[self.cur_batch * self.batch_size + j]][self.cur_step + k][0]
                for j in range(self.batch_size)
            ] for k in range(steps + 1)
        ]
        velocity_y = [
            [
                self.Data[self.sim_path[self.cur_batch * self.batch_size + j]][self.cur_step + k][1]
                for j in range(self.batch_size)
            ] for k in range(steps + 1)
        ]
        velocity_x = [
            [
                self.Data[self.sim_path[self.cur_batch * self.batch_size + j]][self.cur_step + k][2]
                for j in range(self.batch_size)
            ] for k in range(steps + 1)
        ]
        return Re, density, velocity_y, velocity_x

    def next_batch(self):
        self.cur_batch = self.cur_batch + 1
        self.cur_step = 0

    def next_step(self, with_step=1):
        if self.cur_step + with_step >= self.num_step - 1:
            return False
        else:
            self.cur_step = self.cur_step + 1
            return True


def to_feature(V, Re):
    return math.stack(
        [
            V[1].vector['x'].x[:-1].values,
            V[1].vector['y'].y[:-1].values,
            math.ones(V[0].shape) * Re
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


def train_step(t_Res, t_D, t_V, multi_step, res, domain):
    corr_value = []
    corr_state = []
    optimizer.zero_grad()
    for i in range(multi_step):
        den_velocity_temp = simulator.step(
            density_in=t_D[0] if i == 0 else corr_state[-1][0],
            velocity_in=t_V[0] if i == 0 else corr_state[-1][1],
            re=t_Res,
            res=res
        )

        data = to_feature(den_velocity_temp, t_Res)
        data /= math.tensor([RefDataset.Data_std[2], RefDataset.Data_std[1], RefDataset.Data_std[3]],
                            channel('channel'))
        input = (data.native(['batch', 'channel', 'y', 'x'])).to(TORCH.get_default_device().ref)
        output = (Net.forward(input))
        output[:, 0] *= RefDataset.Data_std[2]
        output[:, 1] *= RefDataset.Data_std[1]
        corr_value += [to_staggered(output.cpu(), domain)]
        corr_state += [[den_velocity_temp[0], den_velocity_temp[1] + corr_value[-1]]]

    loss = (torch.nn.MSELoss(reduction='sum'))
    loss_velocity_x = 0
    for i in range(multi_step):
        loss_velocity_x = loss_velocity_x + loss(
            corr_state[i][1].vector['x'].values.native(('batch', 'y', 'x')),
            t_V[i + 1].vector['x'].values.native(('batch', 'y', 'x'))
        ) / RefDataset.Data_std[2]

    loss_velocity_y = 0
    for i in range(multi_step):
        loss_velocity_y = loss_velocity_y + loss(
            corr_state[i][1].vector['y'].values.native(('batch', 'y', 'x')),
            t_V[i + 1].vector['y'].values.native(('batch', 'y', 'x'))
        ) / RefDataset.Data_std[1]

    total_loss = (loss_velocity_x + loss_velocity_y) / multi_step
    total_loss.backward()
    optimizer.step()
    return total_loss

params = {}
parser = argparse.ArgumentParser(description="Parameter parser.")
parser.add_argument('-r', '--res', type=int, default=32, help="Resolution of x axis")
parser.add_argument('-l', '--len', type=int, default=100, help="Length of x axis")
parser.add_argument('--steps', type=int, default=500, help="Total steps")
parser.add_argument('--with_steps', type=int, default=4, help="Consider n steps at a time")
parser.add_argument('--initial_step', type=int, default=1000, help="Initial step")
parser.add_argument('-s', '--sim', type=int, default=6, help="Number of simulation")
parser.add_argument('-e', '--epoch', type=int, default=100, help="Number of epoch")
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--model_path', type=str, default=None, help="Path of model")
parser.add_argument('--device', type=str, default='CPU', help="Device for training")
parser.add_argument('--input', type=str, default='./Reference-Data', help="Path for input data")
parser.add_argument('--output', type=str, default='./model/SOL.pth', help="Path for output model")
args = parser.parse_args()
params.update(vars(args))

TORCH.set_default_device(params['device'])

num_sim = params['sim'] // params['batch_size'] * params['batch_size']
num_batch = num_sim // params['batch_size']
domain = Domain(y=params['res'] * 2, x=params['res'], bounds=Box[0:params['len'] * 2, 0:params['len']], boundaries=OPEN)

simulator = KarmanFlow(domain=domain)
checkpoint = None
Net = Mynet()
if params['model_path'] is not None:
    checkpoint = torch.load(params['model_path'], map_location=TORCH.get_default_device().ref)
    Net.load_state_dict(checkpoint['model'])
Net.to(TORCH.get_default_device().ref)

cur_lr = 0.001
optimizer = optim.Adam(Net.parameters(), lr=cur_lr)
if params['model_path'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])

RefDataset = SOL_Dataset(domain, num_sim, params['input'], params['initial_step'],
                         params['steps'], params['batch_size'], True)
print("Finish read data")

if not os.path.exists(os.path.dirname(params['output'])):
    os.mkdir(os.path.dirname(params['output']))

for i in range(params['epoch']):
    cur_lr = get_lr(i, cur_lr)
    optimizer = optim.Adam(Net.parameters(), lr=cur_lr)
    RefDataset.reboot()
    for j in range(num_batch):
        count = 0
        total_loss = 0
        while 1:
            count = count + 1
            Res, nD, nVy, nVx = RefDataset.Get_Data(params['with_steps'])
            t_Res = math.tensor(Res, math.batch('batch'))
            t_D = [
                domain.scalar_grid(math.tensor(nD[k], batch('batch'), spatial('y,x')))
                for k in range(params['with_steps'] + 1)
            ]
            t_V = [
                domain.staggered_grid(
                    math.stack(
                        [
                            math.tensor(nVy[k], batch('batch'), spatial('y,x')),
                            math.tensor(nVx[k], batch('batch'), spatial('y,x'))
                        ], channel('vector')
                    )
                )
                for k in range(params['with_steps'] + 1)
            ]

            total_loss = total_loss + train_step(t_Res, t_D, t_V, params['with_steps'], params['res'], domain)
            if not RefDataset.next_step(params['with_steps']):
                break
            if count != 0 and count % 10 == 0:
                print(f"The mean loss of steps{count - 9}-steps{count} for epoch{i},batch{j} is {total_loss / 10}")
                total_loss = 0

        RefDataset.next_batch()
    if i % 5 == 4: torch.save({'model': Net.state_dict(), 'optimizer': optimizer.state_dict()}, params['output'])

torch.save({'model': Net.state_dict(), 'optimizer': optimizer.state_dict()}, params['output'])
