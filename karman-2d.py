import os, argparse, pickle, random, distutils.dir_util
from phi.physics._boundaries import Domain, OPEN
from phi.torch.flow import *

random.seed(0)
np.random.seed(0)

from PIL import Image  # for writing PNGs


def save_img(array, scale, name):
    assert tuple.__len__(array.shape) == 2, 'cannot save as an image of {}'.format(array.shape)
    ima = np.reshape(array, [array.shape[0], array.shape[1]])  # remove channel dimension, 2d
    ima = ima[::-1, :]  # flip along y
    image = Image.fromarray(np.asarray(ima * scale, dtype='i'))
    print("\tWriting image: " + name)
    image.save(name)


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

        # apply bc
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


params = {}
parser = argparse.ArgumentParser(description="Parameter parser.")
parser.add_argument('-r', '--res', type=int, default=32, help="Resolution of x axis")
parser.add_argument('-l', '--len', type=int, default=100, help="Length of x axis")
parser.add_argument('--re', type=int, default=1e6, help="Reynolds numbers")
parser.add_argument('--steps', type=int, default=1500, help="Total steps")
parser.add_argument('--device', type=str, default="CPU", help="Device for training")
parser.add_argument('-o', '--output', type=str, default="./Data", help="Output dir")
args = parser.parse_args()
params.update(vars(args))

TORCH.set_default_device(params['device'])
scene = Scene.create(parent_directory=params['output'])

with open(os.path.normpath(scene.path) + '/params.pickle', 'wb') as f:
    pickle.dump(params, f)

domain = Domain(y=params['res'] * 2, x=params['res'], bounds=Box[0:params['len'] * 2, 0:params['len']], boundaries=OPEN)
d0 = domain.scalar_grid(0)

vv = np.ones(domain.staggered_grid().vector['y'].shape.sizes)  # warm start - initialize flow to 1 along y everywhere
uu = np.zeros(domain.staggered_grid().vector['x'].shape.sizes)
uu[uu.shape[0] // 2 + 10:uu.shape[0] // 2 + 20,
uu.shape[1] // 2 - 2:uu.shape[1] // 2 + 2] = 1.0  # modify x, poke sideways to trigger instability
v0 = domain.staggered_grid(
    math.stack([math.tensor(vv, spatial('y, x')), math.tensor(uu, spatial('y, x'))], channel('vector')))

simulator = KarmanFlow(domain=domain)
density, velocity = d0, v0

for i in range(1, params['steps']):
    density, velocity = simulator.step(density, velocity, re=params['re'], res=params['res'])

    scene.write(
        data={
            'dens': density,
            'velo': velocity,
        },
        frame=i
    )

    thumb_path = os.path.normpath(scene.path).replace(os.path.basename(scene.path),
                                                      "thumb/{}".format(os.path.basename(scene.path)))
    distutils.dir_util.mkpath(thumb_path)
    save_img(density.data.numpy(density.values.shape.names), 10000.,
             thumb_path + "/dens_{:06d}.png".format(i))  # shape: [cy, cx]
    save_img(velocity.vector['x'].data.numpy(velocity.vector['x'].values.shape.names), 40000.,
             thumb_path + "/velU_{:06d}.png".format(i))  # shape: [cy, cx+1]
    save_img(velocity.vector['y'].data.numpy(velocity.vector['y'].values.shape.names), 40000.,
             thumb_path + "/velV_{:06d}.png".format(i))  # shape: [cy+1, cx]
