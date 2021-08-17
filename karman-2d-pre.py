import phi.field
import os, argparse, pickle, random
import scipy.sparse.linalg
from phi.physics._boundaries import Domain, OPEN
from phi.torch.flow import *

random.seed(0)
np.random.seed(0)


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


def indexOfcells(cells, bnd=2):
    num, index = 0, np.ones(cells.shape.sizes, dtype=int) * -1
    for j in range(bnd, cells.shape.sizes[0] - bnd):
        for i in range(bnd, cells.shape.sizes[1] - bnd):
            index[j][i] = num
            num = num + 1
    return num, index


def indexOffaces(pre_index, bnd=1, dim=2):
    num, index = [0] * 2, [np.ones(pre_index.shape, dtype=int) * -1 for _ in range(dim)]
    for j in range(bnd, pre_index.shape[0] - bnd):
        for i in range(bnd, pre_index.shape[1] - bnd):
            if pre_index[j][i] > -1 or pre_index[j][i - 1] > -1:
                index[0][j][i] = num[0]
                num[0] = num[0] + 1
            if pre_index[j][i] > -1 or pre_index[j - 1][i] > -1:
                index[1][j][i] = num[1]
                num[1] = num[1] + 1
    return num, index


def Generate_W(param_list):
    jj, ii, h_num, h_index, l_num, l_index, h_corrv, scale = param_list
    D = 4
    scale = scale
    wrow, wcol, wdata, RHS = [], [], [], []
    if h_index[0][jj][ii] > -1:
        x, y = ii / scale, (jj + 0.5) / scale
        l_i, l_j = int(x), int(y - 0.5)
        fx, fy = x - l_i, y - 0.5 - l_j
        wrow += [(h_index[0][jj][ii] * D + 0, h_index[0][jj][ii])]
        wrow += [(h_index[0][jj][ii] * D + 1, h_index[0][jj][ii])]
        wrow += [(h_index[0][jj][ii] * D + 2, h_index[0][jj][ii])]
        wrow += [(h_index[0][jj][ii] * D + 3, h_index[0][jj][ii])]
        w, c = 0, 0
        if l_index[0][l_j][l_i] > -1:
            wcol += [(h_index[0][jj][ii] * D + 0, l_index[0][l_j][l_i])]
            wdata += [[h_index[0][jj][ii] * D + 0, (1 - fx) * (1 - fy)]]
            w += wdata[-1][1]
            c += 1
        if l_index[0][l_j][l_i + 1] > -1:
            wcol += [(h_index[0][jj][ii] * D + 1, l_index[0][l_j][l_i + 1])]
            wdata += [[h_index[0][jj][ii] * D + 1, fx * (1 - fy)]]
            w += wdata[-1][1]
            c += 1
        if l_index[0][l_j + 1][l_i] > -1:
            wcol += [(h_index[0][jj][ii] * D + 2, l_index[0][l_j + 1][l_i])]
            wdata += [[h_index[0][jj][ii] * D + 2, (1 - fx) * fy]]
            w += wdata[-1][1]
            c += 1
        if l_index[0][l_j + 1][l_i + 1] > -1:
            wcol += [(h_index[0][jj][ii] * D + 3, l_index[0][l_j + 1][l_i + 1])]
            wdata += [[h_index[0][jj][ii] * D + 3, fx * fy]]
            w += wdata[-1][1]
            c += 1

        for k in range(c): wdata[-k - 1][1] /= w
        RHS += [(h_index[0][jj][ii], h_corrv[0][jj][ii])]

    h_offset, l_offset = h_num[0], l_num[0]
    if h_index[1][jj][ii] >= 0:
        x, y = (ii + 0.5) / scale, jj / scale
        l_i, l_j = int(x - 0.5), int(y)
        fx, fy = x - l_i - 0.5, y - l_j
        wrow += [((h_index[1][jj][ii] + h_offset) * D + 0, h_index[1][jj][ii] + h_offset)]
        wrow += [((h_index[1][jj][ii] + h_offset) * D + 1, h_index[1][jj][ii] + h_offset)]
        wrow += [((h_index[1][jj][ii] + h_offset) * D + 2, h_index[1][jj][ii] + h_offset)]
        wrow += [((h_index[1][jj][ii] + h_offset) * D + 3, h_index[1][jj][ii] + h_offset)]
        w, c = 0, 0
        if l_index[1][l_j][l_i] > -1:
            wcol += [((h_index[1][jj][ii] + h_offset) * D + 0, l_index[1][l_j][l_i] + l_offset)]
            wdata += [[(h_index[1][jj][ii] + h_offset) * D + 0, (1 - fx) * (1 - fy)]]
            w += wdata[-1][1]
            c += 1
        if l_index[1][l_j][l_i + 1] > -1:
            wcol += [((h_index[1][jj][ii] + h_offset) * D + 1, l_index[1][l_j][l_i + 1] + l_offset)]
            wdata += [[(h_index[1][jj][ii] + h_offset) * D + 1, fx * (1 - fy)]]
            w += wdata[-1][1]
            c += 1
        if l_index[1][l_j + 1][l_i] > -1:
            wcol += [((h_index[1][jj][ii] + h_offset) * D + 2, l_index[1][l_j + 1][l_i] + l_offset)]
            wdata += [[(h_index[1][jj][ii] + h_offset) * D + 2, (1 - fx) * fy]]
            w += wdata[-1][1]
            c += 1
        if l_index[1][l_j + 1][l_i + 1] > -1:
            wcol += [((h_index[1][jj][ii] + h_offset) * D + 3, l_index[1][l_j + 1][l_i + 1] + l_offset)]
            wdata += [[(h_index[1][jj][ii] + h_offset) * D + 3, fx * fy]]
            w += wdata[-1][1]
            c += 1

        for k in range(c): wdata[-k - 1][1] /= w
        RHS += [(h_index[1][jj][ii] + h_offset, h_corrv[1][jj][ii])]

    return wrow, wcol, wdata, RHS


import itertools, multiprocessing


def solveLF(p_corr, d_l, d_h, v_h, beta, scale):
    h_num_cell, h_index_cell = indexOfcells(d_h, bnd=2 * scale)
    l_num_cell, l_index_cell = indexOfcells(d_l, bnd=2)

    h_num_face, h_index_face = indexOffaces(h_index_cell)
    l_num_face, l_index_face = indexOffaces(l_index_cell)

    num_row = sum(h_num_face)
    num_col = sum(l_num_face)

    W_row = np.zeros(num_row * 4, dtype=int)
    W_col = np.zeros(num_row * 4, dtype=int)
    W_data = np.zeros(num_row * 4, dtype=float)
    right_cR = np.zeros((num_row, 1), dtype=float)
    param_list = list(
        itertools.product(range(h_index_cell.shape[0]), range(h_index_cell.shape[1]), [h_num_face], [h_index_face],
                          [l_num_face], [l_index_face], [v_h], [scale]))
    with multiprocessing.Pool(multiprocessing.cpu_count() + 1) as f:
        returns = f.map(Generate_W, param_list)
        for wrow, wcol, wdata, RHS in returns:
            for a in wrow: W_row[a[0]] = a[1]
            for a in wcol: W_col[a[0]] = a[1]
            for a in wdata: W_data[a[0]] = a[1]
            for a in RHS: right_cR[a[0]] = a[1]

    matW = scipy.sparse.csr_matrix((W_data, (W_row, W_col)), shape=(num_row, num_col), dtype=float)

    D = 2
    M_row = np.zeros(num_col * 2, dtype=int)
    M_col = np.zeros(num_col * 2, dtype=int)
    M_data = np.zeros(num_col * 2, dtype=float)
    right_pre = np.zeros((num_col, 1), dtype=float)
    for j in range(l_index_cell.shape[0]):
        for i in range(l_index_cell.shape[1]):
            if l_index_face[0][j][i] > -1:
                right_pre[l_index_face[0][j][i]] = p_corr[0][j][i] * 2
                M_row[l_index_face[0][j][i] * D + 0] = l_index_face[0][j][i]
                M_row[l_index_face[0][j][i] * D + 1] = l_index_face[0][j][i]
                if l_index_cell[j][i] > -1:
                    M_col[l_index_face[0][j][i] * D + 0] = l_index_cell[j][i]
                    M_data[l_index_face[0][j][i] * D + 0] = 1.0
                if l_index_cell[j][i - 1] > -1:
                    M_col[l_index_face[0][j][i] * D + 1] = l_index_cell[j][i - 1]
                    M_data[l_index_face[0][j][i] * D + 1] = -1.0
            l_offset = l_num_face[0]
            if l_index_face[1][j][i] > -1:
                right_pre[l_index_face[1][j][i] + l_offset] = p_corr[1][j][i] * 2
                M_row[(l_index_face[1][j][i] + l_offset) * D + 0] = l_index_face[1][j][i] + l_offset
                M_row[(l_index_face[1][j][i] + l_offset) * D + 1] = l_index_face[1][j][i] + l_offset
                if l_index_cell[j][i] > -1:
                    M_col[(l_index_face[1][j][i] + l_offset) * D + 0] = l_index_cell[j][i]
                    M_data[(l_index_face[1][j][i] + l_offset) * D + 0] = 1.0
                if l_index_cell[j - 1][i] > -1:
                    M_col[(l_index_face[1][j][i] + l_offset) * D + 1] = l_index_cell[j - 1][i]
                    M_data[(l_index_face[1][j][i] + l_offset) * D + 1] = -1.0

    matM = scipy.sparse.csr_matrix((M_data, (M_row, M_col)), shape=(num_col, l_num_cell), dtype=float)
    mat2I = scipy.sparse.identity(num_col, dtype=float) * 2
    matWTW_inv = scipy.sparse.linalg.inv((matW.transpose()).dot(matW) + (mat2I * beta if beta > 0 else 0))
    A = (matM.transpose()).dot(matWTW_inv).dot(matM)
    b = (matM.transpose()).dot(matWTW_inv).dot((matW.transpose()).dot(right_cR) + (beta * right_pre if beta > 0 else 0))
    X, cginfo = scipy.sparse.linalg.cg(A, b)
    X = X.reshape((l_num_cell, 1))
    c_S = matWTW_inv.dot((matW.transpose()).dot(right_cR) - matM.dot(X))
    corrV = np.zeros((l_index_cell.shape[0] + 1, l_index_cell.shape[1]), dtype=float)
    corrU = np.zeros((l_index_cell.shape[0], l_index_cell.shape[1] + 1), dtype=float)
    for j in range(l_index_cell.shape[0]):
        for i in range(l_index_cell.shape[1]):
            if l_index_face[0][j][i] > -1:
                corrU[j][i] = c_S[l_index_face[0][j][i]]
            if l_index_face[1][j][i] > -1:
                corrV[j][i] = c_S[l_index_face[1][j][i] + l_num_face[0]]

    return domain_l.staggered_grid(
        math.stack(
            [
                math.tensor(corrV, spatial('y,x')),
                math.tensor(corrU, spatial('y,x'))
            ]
            , math.channel('vector')
        )
    )


def correction(simulator, pre_corr, density_l, density_h, velocity_l, velocity_h, re, res, beta, scale, dt=1.0):
    obstacles = [Obstacle(Sphere(center=[50, 50], radius=10))]
    temp_d, temp_v = simulator.step(density_l, velocity_l, re, res)

    vh_diff = velocity_h - temp_v.at(domain_h.staggered_grid())
    vh_diff, _ = fluid.make_incompressible(vh_diff, obstacles)
    if pre_corr is not None:
        v_pre_corr = pre_corr
    else:
        v = np.zeros(velocity_l.vector['y'].shape.sizes)
        u = np.zeros(velocity_l.vector['x'].shape.sizes)
        v_pre_corr = domain_l.staggered_grid(
            math.stack([math.tensor(v, spatial('y, x')), math.tensor(u, spatial('y, x'))], channel('vector')))
    v_corr = solveLF([v_pre_corr.vector['x'].values.numpy('y,x'), v_pre_corr.vector['y'].values.numpy('y,x')], temp_d,
                     density_h, [vh_diff.vector['x'].values.numpy('y,x'), vh_diff.vector['y'].values.numpy('y,x')],
                     beta / dt, scale)

    return v_corr, temp_d, temp_v + v_corr, temp_v


params = {}
parser = argparse.ArgumentParser(description="Parameter parser.")
parser.add_argument('-r', '--res', type=int, default=32, help="Resolution of x axis")
parser.add_argument('-l', '--len', type=int, default=100, help="Length of x axis")
parser.add_argument('--re', type=int, default=1e6, help="Reynolds numbers")
parser.add_argument('--steps', type=int, default=1500, help="Total steps")
parser.add_argument('--scale', type=int, default=4, help="Minization scale")
parser.add_argument('-o', '--output', type=str, default="./Pre-Data", help="Output dir")
parser.add_argument('--beta', type=float, default=0, help="Penalty factor")
args = parser.parse_args()
params.update(vars(args))

scene = Scene.create(parent_directory=params['output'])

with open(os.path.normpath(scene.path) + '/params.pickle', 'wb') as f:
    pickle.dump(params, f)

domain_h = Domain(y=params['res'] * 2 * params['scale'], x=params['res'] * params['scale'],
                  bounds=Box[0:params['len'] * 2, 0:params['len']],
                  boundaries=OPEN)
domain_l = Domain(y=params['res'] * 2, x=params['res'],
                  bounds=Box[0:params['len'] * 2, 0:params['len']], boundaries=OPEN)

d0 = domain_h.scalar_grid(0)
vv = np.ones(domain_h.staggered_grid().vector['y'].shape.sizes)  # warm start - initialize flow to 1 along y everywhere
uu = np.zeros(domain_h.staggered_grid().vector['x'].shape.sizes)
uu[uu.shape[0] // 2 + 10:uu.shape[0] // 2 + 20,
uu.shape[1] // 2 - 2:uu.shape[1] // 2 + 2] = 1.0  # modify x, poke sideways to trigger instability
v0 = domain_h.staggered_grid(
    math.stack([math.tensor(vv, spatial('y, x')), math.tensor(uu, spatial('y, x'))], channel('vector')))
simulator_h = KarmanFlow(domain=domain_h)
simulator_l = KarmanFlow(domain=domain_l)

density_h, velocity_h = d0, v0
density_l, velocity_l = d0.at(domain_l.scalar_grid()), v0.at(domain_l.staggered_grid())

pre_corr = None

for i in range(1, params['steps']):
    density_h, velocity_h = simulator_h.step(density_h, velocity_h, re=params['re'],
                                             res=params['res'] * params['scale'])
    pre_corr, density_l, velocity_l, p_velocity_l = \
        correction(simulator_l, pre_corr, density_l, density_h, velocity_l,
                   velocity_h, params['re'], params['res'], params['beta'],
                   params['scale'])

    scene.write(
        data={
            'source_velo': p_velocity_l,
            'corr_velo': pre_corr
        },
        frame=i
    )
