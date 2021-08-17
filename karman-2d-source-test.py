import math
import os, pickle, glob, argparse
import phi.field
from phi.physics._boundaries import Domain, OPEN
from phi.torch.flow import *


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
        if self.need_downsample:
            return os.path.dirname(file) + '/ds/' + os.path.basename(file)
        else:
            return file

    def Get_Data(self, sim, step):
        Re = self.Res[self.sim_path[sim]]
        density = self.Data[self.sim_path[sim]][step][0]
        velocity_y = self.Data[self.sim_path[sim]][step][1]
        velocity_x = self.Data[self.sim_path[sim]][step][2]
        return Re, density, velocity_y, velocity_x


def get_error(simulator, t_Re, s_D, s_V, r_V, res):
    sour_state = simulator.step(density_in=s_D, velocity_in=s_V, re=t_Re, res=res)

    loss = torch.nn.L1Loss(reduction='mean')

    loss_velocity_x = loss(
        sour_state[1].vector['x'].values.native(('batch', 'y', 'x')),
        r_V.vector['x'].values.native(('batch', 'y', 'x'))
    )
    loss_velocity_y = loss(
        sour_state[1].vector['y'].values.native(('batch', 'y', 'x')),
        r_V.vector['y'].values.native(('batch', 'y', 'x'))
    )
    total_loss = loss_velocity_x + loss_velocity_y
    return total_loss, sour_state[0], sour_state[1]


params = {}
parser = argparse.ArgumentParser(description="Parameter parser.")
parser.add_argument('-r', '--res', type=int, default=32, help="Resolution of x axis")
parser.add_argument('-l', '--len', type=int, default=100, help="Length of x axis")
parser.add_argument('--steps', type=int, default=500, help="Total steps")
parser.add_argument('-s', '--sim', type=int, default=5, help="Number of simulation")
parser.add_argument('--initial_step', type=int, default=1000, help="Initial step")
parser.add_argument('--input', type=str, default='./Test-Data', help="Path for input data")
args = parser.parse_args()
params.update(vars(args))
domain = Domain(y=params['res'] * 2, x=params['res'], bounds=Box[0:params['len'] * 2, 0:params['len']], boundaries=OPEN)
RefDataset = Dataset(domain, params['sim'], params['input'], params['initial_step'], params['steps'], True)
print("Finish data reading")
simulator = KarmanFlow(domain=domain)
error = []

for i in range(params['sim']):
    total_error = 0
    Re, d0, V0y, V0x = RefDataset.Get_Data(i, 0)
    density = domain.scalar_grid(math.tensor(d0, spatial('y,x')))
    velocity = domain.staggered_grid(
        math.stack(
            [
                math.tensor(V0y, spatial('y,x')),
                math.tensor(V0x, spatial('y,x'))
            ], channel('vector')
        )
    )
    for j in range(params['steps'] - 1):
        R, D, RVy, RVx = RefDataset.Get_Data(i, j + 1)
        ref_density = domain.scalar_grid(math.tensor(D, spatial('y,x')))
        ref_velocity = domain.staggered_grid(
            math.stack(
                [
                    math.tensor(RVy, spatial('y,x')),
                    math.tensor(RVx, spatial('y,x'))
                ], channel('vector')
            )
        )
        one_error, density, velocity = get_error(simulator, Re, density, velocity, ref_velocity, params['res'])
        total_error = total_error + one_error
    error += [total_error / (params['steps'] - 1)]
    print(f"The mean error for sim{i} is {error[-1]}")

print(np.mean(error))
