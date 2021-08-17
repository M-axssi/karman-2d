import math
import os, argparse, pickle, glob, distutils.dir_util
import phi.field
from phi.physics._boundaries import Domain, OPEN
from phi.torch.flow import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn


def normalization(data):
    _range = np.max(data) - np.min(data)
    mid = (np.max(data) + np.min(data)) / 2
    return (data - mid) / _range


def save_img(array, name):
    if tuple.__len__(array.shape) == 3:
        ima = np.reshape(array, [array.shape[1], array.shape[2]])  # remove channel dimension, 2d
    else:
        ima = array
    ima = normalization(ima[::-1, :])  # flip along y
    plt.axis('off')
    plt.imshow(np.asarray(ima), cmap="RdYlBu")
    print("\tWriting image: " + name)
    plt.savefig(name, bbox_inches='tight', pad_inches=0)


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
             make_output_divfree=True):
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


class Dataset:
    def __init__(self, domain, num_sim, dirpath, begin_step, num_step, need_downsample=True):
        self.sim_path = sorted(glob.glob(dirpath + "/sim_0*"))[0:num_sim]
        self.density_path = [sorted(glob.glob(asim + "/dens_0*.npz"))[begin_step - 1:begin_step + num_step] for asim in
                             self.sim_path]
        self.velocity_path = [sorted(glob.glob(asim + "/velo_0*.npz"))[begin_step - 1:begin_step + num_step] for asim in
                              self.sim_path]
        self.domain = domain
        self.need_downsample = need_downsample
        self.num_sim = num_sim
        self.num_step = num_step
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

    def Get_Data(self, sim, step):
        Re = self.Res[self.sim_path[sim]]
        density = self.Data[self.sim_path[sim]][step][0]
        velocity_y = self.Data[self.sim_path[sim]][step][1]
        velocity_x = self.Data[self.sim_path[sim]][step][2]
        return Re, density, velocity_y, velocity_x


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
                math.tensor(F.pad(output[:, 1], (0, 0, 0, 1), mode='constant', value=0), math.batch('batch'),
                            spatial('y,x')),
                math.tensor(F.pad(output[:, 0], (0, 1, 0, 0), mode='constant', value=0), math.batch('batch'),
                            spatial('y,x'))
            ]
            , math.channel('vector')
        )
    )


Data_std = [2.4450324, 0.3312727, 0.19495082, 1732512.6262166172]


def corr_simulate(t_Re, s_D, s_V, type):
    sour_state = simulator.step(density_in=s_D, velocity_in=s_V, re=t_Re, res=params['res'])
    if type == "Source":
        return sour_state[0], sour_state[1]
    else:
        input = to_feature(sour_state, t_Re)
        input /= math.tensor([Data_std[2], Data_std[1], Data_std[3]],
                             channel('channel'))
        output = Net.forward(input.native(['batch', 'channel', 'y', 'x']))
        output[:, 0] *= Data_std[2]
        output[:, 1] *= Data_std[1]
        corr_value = to_staggered(output, domain)
        corr_velocity = sour_state[1] + corr_value
        return sour_state[0], corr_velocity


params = {}
parser = argparse.ArgumentParser(description="Parameter parser.")
parser.add_argument('-r', '--res', type=int, default=32, help="Resolution of x axis")
parser.add_argument('-l', '--len', type=int, default=100, help="Length of x axis")
parser.add_argument('-s', '--sim', type=int, default=5, help="Number of simulation")
parser.add_argument('--steps', type=int, default=200, help="Total steps")
parser.add_argument('--initial_step', type=int, default=1000)
parser.add_argument('--model_path', type=str, default=None, help="Path of model")
parser.add_argument('--output', type=str, default='./img_result', help="Path of model")
parser.add_argument('--type', type=str, default='Ref', help="Can choose Ref , Source or Model")
parser.add_argument('--device', type=str, default='CPU', help="Device for training")
parser.add_argument('--input', type=str, default='./Test-Data', help="Path for input data")
parser.add_argument('--mark', type=str, default='0', help="Which sims will be write")
args = parser.parse_args()
params.update(vars(args))

TORCH.set_default_device(params['device'])
domain = Domain(y=params['res'] * 2, x=params['res'], bounds=Box[0:params['len'] * 2, 0:params['len']], boundaries=OPEN)
simulator = KarmanFlow(domain=domain)
Net = Mynet()
if params['model_path'] is not None:
    Net.load_state_dict(torch.load(params['model_path'], map_location=TORCH.get_default_device().ref)['model'])
    Net.to(TORCH.get_default_device().ref)

TestDataset = Dataset(domain, params['sim'], params['input'], params['initial_step'], params['steps'] + 1, True)
print("Finish data reading")

write_list = list(map(int, params['mark'].split(',')))
with torch.no_grad():
    for i in write_list:
        total_error = 0
        Re, d0, V0y, V0x = TestDataset.Get_Data(i, 0)
        density = domain.scalar_grid(math.tensor(d0, spatial('y,x')))
        velocity = domain.staggered_grid(
            math.stack(
                [
                    math.tensor(V0y, spatial('y,x')),
                    math.tensor(V0x, spatial('y,x'))
                ], channel('vector')
            )
        )
        for j in range(params['steps']):
            if params['type'] == "Ref":
                R, D, RVy, RVx = TestDataset.Get_Data(i, j + 1)
                density = domain.scalar_grid(math.tensor(D, spatial('y,x')))
                velocity = domain.staggered_grid(
                    math.stack(
                        [
                            math.tensor(RVy, spatial('y,x')),
                            math.tensor(RVx, spatial('y,x'))
                        ], channel('vector')
                    )
                )
            else:
                density, velocity = corr_simulate(Re, density, velocity, params['type'])

            thumb_path = params['output'] + "/sim_{:06d}".format(i)
            distutils.dir_util.mkpath(thumb_path)
            distutils.dir_util.mkpath(thumb_path + "/density")
            distutils.dir_util.mkpath(thumb_path + "/velU")
            distutils.dir_util.mkpath(thumb_path + "/velV")
            save_img(density.data.numpy(density.values.shape.names),
                     thumb_path + "/density/dens_{:06d}.png".format(j + 1))
            save_img(velocity.vector['x'].data.numpy(velocity.vector['x'].values.shape.names),
                     thumb_path + "/velU/velU_{:06d}.png".format(j + 1))
            save_img(velocity.vector['y'].data.numpy(velocity.vector['y'].values.shape.names),
                     thumb_path + "/velV/velV_{:06d}.png".format(j + 1))
