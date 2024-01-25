#!/usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: bsUtils.py 
@time: 2023/01/31
@contact: miyuan@ruc.edu.cn
@site:  
@software: PyCharm 

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓          ┏┓
            ┏┛┻━━━━━━━━━━┛┻┓
            ┃      ☃       ┃
            ┃    ┳┛  ┗┳    ┃
            ┃       ┻      ┃
            ┗━━━┓      ┏━━━┛
                ┃      ┗━━━━━━━━┓
                ┃  神兽保佑       ┣┓
                ┃　永无BUG！      ┏┛
                ┗━┓┓┏━━━━━━━━┳┓┏┛
                  ┃┫┫        ┃┫┫
                  ┗┻┛        ┗┻┛ 
"""
import torch
import xarray as xr
import numpy as np
import einops as eo

import skfem
from sklearn_extra.cluster import KMedoids

from dataclasses import dataclass, replace
from functools import cached_property
from itertools import combinations
from typing import Optional

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay
import sys

sys.path.append("..")


def inside_angle(a, b, c):
    """Compute the inside angle of a right triangle at a with a right angle at b."""

    opposite = np.linalg.norm(b - c)
    hypotenuse = np.linalg.norm(a - c)

    # Due to floating point errors, this ratio can actually become larger than 1, so we
    # cap it at 1 to avoid a warning
    ratio = min(opposite / hypotenuse, 1.0)
    return np.arcsin(ratio)


def project_onto(p, line_p):
    assert len(line_p) == 2, "Can only project onto lines"
    a, b = line_p
    ab = b - a
    ap = p - a
    ab_norm = ab / np.linalg.norm(ab)
    return a + (np.inner(ap, ab_norm)) * ab_norm


class CellPredicate:
    def __call__(
            self, cell_idx: int, cell: list[int], boundary_faces: list[list[int]]
    ) -> bool:
        """Decide if a cell fulfills the predicate.
        Arguments
        ---------
        cell_idx
            Index of the cell in tri.simplices
        cell
            Node indices of the cell vertices
        boundary_faces
            Faces of the cell that are on the boundary of the mesh
        """

        raise NotImplementedError()


def select_boundary_mesh_cells(tri: Delaunay, predicate: CellPredicate) -> np.ndarray:
    """Delete mesh cells from the boundary of a mesh that fulfill a predicate.
    This function implements the iterative boundary filtering algorithm described in
    Appendix F.
    Returns
    -------
    A mask over `tri.points` that selects nodes to delete
    """

    # Put the vertex indices of a boundary edge in a canonical order
    e = lambda nodes: tuple(sorted(nodes))

    cells = tri.simplices
    n_cells = len(cells)
    n_vertices = cells.shape[-1]
    n_face_vertices = n_vertices - 1
    adjacent = [[] for i in range(n_cells)]
    node_sets = [set(cells[i]) for i in range(n_cells)]
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            if len(node_sets[i] & node_sets[j]) == n_face_vertices:
                adjacent[i].append(j)
                adjacent[j].append(i)

    delete_cells = np.zeros(len(cells), dtype=bool)
    visited = np.zeros(len(cells), dtype=bool)
    boundary_faces = set(e(nodes) for nodes in tri.convex_hull)
    faces_on_boundary = np.array(
        [
            sum(
                [
                    e(nodes) in boundary_faces
                    for nodes in combinations(cell, n_face_vertices)
                ]
            )
            for cell in cells
        ]
    )
    boundary_stack = [
        (i, cell) for i, cell in enumerate(cells) if faces_on_boundary[i] > 0
    ]
    while len(boundary_stack) > 0:
        i, cell = boundary_stack.pop()
        visited[i] = True

        # Separate boundary and interior nodes
        cell_boundary_faces = [
            nodes
            for nodes in combinations(cell, n_face_vertices)
            if e(nodes) in boundary_faces
        ]

        if predicate(i, cell, cell_boundary_faces):
            delete_cells[i] = True

            # If a mesh cell is filtered, mark all its faces as boundary (even the
            # exterior ones; they don't matter and it simplifies the code)
            for nodes in combinations(cell, n_face_vertices):
                boundary_faces.add(e(nodes))

            # The adjacent interior cells have now become boundary cells and need
            # to be inspected as well
            for j in adjacent[i]:
                if delete_cells[j] or visited[j]:
                    continue

                boundary_stack.append((j, cells[j]))

    return delete_cells


@dataclass
class BoundaryAnglePredicate(CellPredicate):
    """Check if a boundary cell is too acute/elongated.
    A cell is flagged as too acute if its interior node is too close to the boundary.
    """

    p: np.ndarray
    epsilon: float = np.pi / 180

    def __call__(self, cell_idx, cell, boundary_faces):
        if len(boundary_faces) != 1:
            # Keep any cell that is on an edge or corner of the domain
            return False

        boundary_nodes = boundary_faces[0]
        interior_node = list(set(cell) - set(boundary_nodes))[0]

        # Project the interior node onto the boundary surface
        interior_p = self.p[interior_node]
        interior_projection = project_onto(interior_p, self.p[list(boundary_nodes)])

        # Compute the inside angle of the right triangle (interior_p, a,
        # interior_projection) for all boundary nodes a
        angles = [
            inside_angle(self.p[a], interior_projection, interior_p)
            for a in boundary_nodes
        ]

        # Minimum inside angle at the boundary points
        min_angle = min(angles)

        if np.isnan(min_angle):
            # I have observed this for some degenerate triangles, so we just filter them
            # out
            return True
        else:
            return min_angle < self.epsilon


def select_acute_boundary_triangles(
        tri: Delaunay, epsilon: float = np.pi / 180
) -> np.ndarray:
    """Select highly acute triangle artifacts from the boundary of a Delaunay
    triangulation.
    It is important to note that multiple (almost, up to float precision) linear points
    (i.e. on a line), can create layers of these acute triangles. This necessitates the
    iterative algorithm below.
    Parameters
    ----------
    tri
        Delaunay triangulation
    epsilon
        Maximum angle to filter in radians
    """

    predicate = BoundaryAnglePredicate(tri.points, epsilon)
    return select_boundary_mesh_cells(tri, predicate)


@dataclass
class Domain:
    """A fixed observation domain consisting of nodes and a mesh.
    Attributes
    ----------
    x : ndarray of size n_nodes x space_dim
        Location of the nodes
    mesh
        A mesh of the nodes. You can optionally pass in a fixed or precomputed mesh.
    fixed_values_mask : boolean ndarray of size n_nodes
        An optional mask that selects nodes where the values are fixed and therefore no
        prediction should be made
    """

    x: np.ndarray
    # You can pass None but after __post_init__ this attribute will always hold a mesh, so
    # it is not marked as Optional
    mesh: skfem.Mesh = None
    fixed_values_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.x.ndim == 2
        assert self.x.shape[1] in (1, 2, 3)

        if self.mesh is None:
            self.mesh = self._generate_mesh()

        self._reorder_vertices()

    def normalize(self) -> "Domain":
        """Normalize the node coordinates to mean 0 and mean length of 1."""
        mean = self.x.mean(axis=0, keepdims=True)
        std = np.linalg.norm(self.x - mean, axis=-1).mean()
        x = (self.x - mean) / std
        mesh = self.mesh
        if mesh is not None:
            mesh = replace(mesh, doflocs=x.T)
        return replace(self, x=x, mesh=mesh)

    def __str__(self):
        return f"<{len(self)} points in {self.dim}D; min={self.x.min():.3f}, max={self.x.max():.3f}>"

    def __len__(self):
        return self.x.shape[0]

    @property
    def dim(self):
        return self.x.shape[1]

    @cached_property
    def basis(self):
        return skfem.CellBasis(self.mesh, self.mesh.elem())

    def _generate_mesh(self):
        doflocs = np.ascontiguousarray(self.x.T)
        if self.dim == 1:
            return skfem.MeshLine(np.sort(doflocs))
        elif self.dim == 2:
            tri = Delaunay(self.x)
            simplices = tri.simplices[~select_acute_boundary_triangles(tri)]
            return skfem.MeshTri(doflocs, np.ascontiguousarray(simplices.T))
        elif self.dim == 3:
            tri = Delaunay(self.x)
            simplices = tri.simplices[~select_acute_boundary_triangles(tri)]
            return skfem.MeshTet(doflocs, np.ascontiguousarray(simplices.T))

    def _reorder_vertices(self):
        # Put cell vertices into some "canonical" order to make it easier for models to
        # generalize
        if self.dim in (2, 3):
            cells = self.mesh.t.T
            vertices = self.x[cells]
            cell_centers = vertices.mean(axis=1)
            # Convert into cell-local coordinates
            vertex_local = vertices - cell_centers[:, None, :]

            # Compute the angle of the vertices' polar coordinates. We do the same thing in
            # both 2 and 3 dimensions which corresponds to projecting 3D points onto the 2D
            # plane first. Is there something better?
            theta = np.arctan2(vertex_local[..., 1], vertex_local[..., 0])

            order = np.argsort(theta, axis=1)
            cells = np.take_along_axis(cells, order, axis=1)

            self.mesh = replace(self.mesh, t=np.ascontiguousarray(cells.T))


from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class MeshConfig:
    """Configuration for the generation of a sparse mesh from a larger set of points.
    Attributes
    ----------
    k
        Number of nodes to select
    epsilon
        Maximum angle of boundary cells to filter out in degrees
    seed
        Random seed for reproducibility
    """

    k: int
    epsilon: float
    seed: int

    def random_state(self):
        return np.random.RandomState(int(self.seed) % 2 ** 32)

    def epsilon_radians(self):
        return self.epsilon * np.pi / 180

    def angle_predicate(self, tri: Delaunay):
        return BoundaryAnglePredicate(tri.points, self.epsilon_radians())


def sample_mesh(
        config: MeshConfig,
        points: np.ndarray,
        predicate_factory: Optional[Callable[[Delaunay], CellPredicate]] = None,
) -> tuple[np.ndarray, Domain]:
    """Create a domain from a subset of points, optionally filtering out some cells.
    Returns
    -------
    Indices of the points that were selected as mesh nodes and the domain
    """
    # Select k sparse observation points uniformly-ish
    km = KMedoids(
        n_clusters=config.k, init="k-medoids++", random_state=config.random_state()
    )
    km.fit(points)
    node_indices = km.medoid_indices_

    # Mesh the points with Delaunay triangulation
    tri = Delaunay(points[node_indices])

    # Filter out mesh boundary cells that are too acute or contain mostly land
    if predicate_factory is not None:
        predicate = predicate_factory(tri)
        filter = select_boundary_mesh_cells(tri, predicate)
        tri.simplices = tri.simplices[~filter]

    # Ensure that every node is in at least one mesh cell
    cell_counts = np.zeros(config.k, dtype=int)
    np.add.at(cell_counts, tri.simplices, 1)
    assert all(cell_counts >= 1)

    mesh = skfem.MeshTri(
        np.ascontiguousarray(tri.points.T), np.ascontiguousarray(tri.simplices.T)
    )
    domain = Domain(tri.points, mesh=mesh)
    return node_indices, domain


class OutOfDomainPredicate(CellPredicate):
    """Filter out all mesh cells that include mostly out-of-domain points."""

    def __init__(self, tri: Delaunay, x: np.ndarray, in_domain: np.ndarray):
        """
        Arguments
        ---------
        tri
            A mesh defined over some points
        x
            A set of "trial points" to check the mesh cells against
        in_domain
            A mask that says which points in `x` are in-domain
        """

        vertices = tri.points[tri.simplices]

        a, b, c = np.split(vertices, 3, axis=-2)
        ab, bc, ca = b - a, c - b, a - c
        ax, bx, cx = x - a, x - b, x - c

        # A point is inside a triangle if all these cross-products have the same sign
        abx = np.cross(ab, ax)
        bcx = np.cross(bc, bx)
        cax = np.cross(ca, cx)
        inside = ((abx * bcx) >= 0) & ((bcx * cax) >= 0) & ((cax * abx) >= 0)

        n_in_domain_in_cell = np.logical_and(inside, in_domain).sum(axis=-1)
        n_out_of_domain_in_cell = np.logical_and(inside, ~in_domain).sum(axis=-1)
        self.filter = n_out_of_domain_in_cell > n_in_domain_in_cell

    def __call__(self, cell_idx, cell, boundary_faces):
        return self.filter[cell_idx]


def default_mesh_config():
    # Arbitrarily chosen, fixed seed
    return MeshConfig(k=1000, epsilon=10.0, seed=803452658411725)


import pytorch_lightning as pl
from pathlib import Path
import subprocess
import xarray as xr
import os


# self.mesh_config = (
#     self.default_mesh_config() if mesh_config is None else mesh_config
# )
# self.mesh_file = self.root / "domain.pt"
# self.stats_file = self.root / "stats.npz"

def _download_raw_data():
    features = [
        ("thetao", "cmems_mod_blk_phy-tem_my_2.5km_P1D-m", ["thetao"]),
        ("uo", "cmems_mod_blk_phy-cur_my_2.5km_P1D-m", ["uo"]),
        ("vo", "cmems_mod_blk_phy-cur_my_2.5km_P1D-m", ["vo"]),
        # ("thetao", "bs-cmcc-tem-rean-d", ["thetao"]),
        # ("uo", "bs-cmcc-cur-rean-d", ["uo"]),
        # ("vo", "bs-cmcc-cur-rean-d", ["vo"]),
    ]
    jobs = [
        (product_id, variables, feature)
        for feature, product_id, variables in features
        # if not (target := xr.open_mfdataset(f"{feature}.nc")).is_file()
    ]
    if len(jobs) == 0:
        return
    for args in jobs:
        download_motu(*args)


def download_motu(product_id: str, variables: list[str], target: str):
    variable_options = []
    for var in variables:
        variable_options.append("--variable")
        variable_options.append(var)

    service_id = 'BLKSEA_MULTIYEAR_PHY_007_004-TDS'
    PRODUCT_TYPE = 'my'
    lat_min = '40.86'
    lat_max = '46.8044'
    lon_min = '27.37'
    lon_max = '41.9626'
    start = f'1993-01-01 00:00:00'
    # [ERROR] 010-7 : The result file size, 3372.0Mb, is too big and shall be less than 2048.0Mb.
    # Please narrow your request.
    mid_1 = f'2003-01-01 00:00:00'
    mid_2 = f'2013-01-01 00:00:00'
    end = f'2021-06-30 23:59:59'
    depth_min = '12.5'
    depth_max = '12.5362'
    USERNAME = 'ymi'
    PASSWORD = 'My12345678'
    OUTDIR = "/mnt/miyuan/AI4Physics/Data/bs"
    if not os.path.exists(OUTDIR):
        print(OUTDIR)
        os.makedirs(OUTDIR)

    OUTNAME = f"{start[:4]}_{mid_1[:4]}_{target}.nc"
    cmd_1 = ["/home/miyuan/anaconda3/envs/py39/bin/python", "-m", "motuclient", "--motu",
           "https://my.cmems-du.eu/motu-web/Motu",
           "--service-id", service_id, "--product-id", product_id,
           "--longitude-min", lon_min, "--longitude-max", lon_max, "--latitude-min", lat_min, "--latitude-max", lat_max,
           "--date-min", start, "--date-max", mid_1, "--depth-min", depth_min, "--depth-max", depth_max,
           *variable_options,
           "--out-dir", OUTDIR, "--out-name", OUTNAME, "--user", USERNAME, "--pwd", PASSWORD,
           # "--config-file",
           # "~/.config/motuclient/motuclient-python.ini",
           ]
    # subprocess.run(cmd, check=True, shell=True)
    # subprocess.Popen(cmd)
    subprocess.run(cmd_1)

    OUTNAME = f"{mid_1[:4]}_{mid_2[:4]}_{target}.nc"
    cmd_2 = ["/home/miyuan/anaconda3/envs/py39/bin/python", "-m", "motuclient", "--motu",
           "https://my.cmems-du.eu/motu-web/Motu",
           "--service-id", service_id, "--product-id", product_id,
           "--longitude-min", lon_min, "--longitude-max", lon_max, "--latitude-min", lat_min, "--latitude-max", lat_max,
           "--date-min", mid_1, "--date-max", mid_2, "--depth-min", depth_min, "--depth-max", depth_max,
           *variable_options,
           "--out-dir", OUTDIR, "--out-name", OUTNAME, "--user", USERNAME, "--pwd", PASSWORD,
           # "--config-file",
           # "~/.config/motuclient/motuclient-python.ini",
           ]
    # subprocess.run(cmd, check=True, shell=True)
    # subprocess.Popen(cmd)
    # subprocess.run(cmd_2)

    OUTNAME = f"{mid_2[:4]}_{end[:4]}_{target}.nc"
    cmd_3 = ["/home/miyuan/anaconda3/envs/py39/bin/python", "-m", "motuclient", "--motu",
           "https://my.cmems-du.eu/motu-web/Motu",
           "--service-id", service_id, "--product-id", product_id,
           "--longitude-min", lon_min, "--longitude-max", lon_max, "--latitude-min", lat_min, "--latitude-max", lat_max,
           "--date-min", mid_2, "--date-max", end, "--depth-min", depth_min, "--depth-max", depth_max,
           *variable_options,
           "--out-dir", OUTDIR, "--out-name", OUTNAME, "--user", USERNAME, "--pwd", PASSWORD,
           # "--config-file",
           # "~/.config/motuclient/motuclient-python.ini",
           ]
    # subprocess.run(cmd, check=True, shell=True)
    # subprocess.Popen(cmd)
    # subprocess.run(cmd_3)
    # query = f"python -m motuclient --motu https://{PRODUCT_TYPE}.cmems-du.eu/motu-web/Motu " \
    #         f"--service-id {service_id} --product-id {product_id} " \
    #         f"--longitude-min {lon_min} --longitude-max {lon_max} --latitude-min {lat_min} --latitude-max {lat_max} " \
    #         f"--date-min {start} --date-max {end} --depth-min {depth_min} --depth-max {depth_max} " \
    #         f"--variable {variable_options[1]}" \
    #         f"--out-dir {OUTDIR} --out-name {OUTNAME} --user {USERNAME} --pwd {PASSWORD}"
    # '--variable thetao'
    # os.system(query)


def _compute_stats(self, t: np.ndarray, u: np.ndarray):
    n_nodes = u.shape[1]
    velocity, temperature = u[..., :2], u[..., 2]

    # For the temperature we compute separate stats for each calendar day because
    # there is a strong periodicity throughout the year. Also we just ignore leap
    # years.
    day = t.astype(int) % 365

    # Count the number of training samples per calendar day
    counts = np.zeros(365)
    np.add.at(counts, day, 1)

    sums = np.zeros(365)
    np.add.at(sums, day, temperature.sum(axis=1))
    daily_temperature_mean = sums / counts / n_nodes

    sums = np.zeros(365)
    np.add.at(
        sums,
        day,
        ((temperature - daily_temperature_mean[day, None]) ** 2).sum(axis=1),
    )
    daily_temperature_std = np.sqrt(sums / counts / n_nodes)

    all_velocities = eo.rearrange(velocity, "t n f -> (t n) f")
    mean = np.concatenate(
        (
            eo.repeat(np.mean(all_velocities, axis=0), "f -> 365 f"),
            daily_temperature_mean[:, None],
        ),
        axis=-1,
    )
    std = np.concatenate(
        (
            eo.repeat(np.std(all_velocities, axis=0), "f -> 365 f"),
            daily_temperature_std[:, None],
        ),
        axis=-1,
    )
    return mean, std


# if __name__ == '__main__':
    # wget ftp://my.cmems-du.eu/Core/BLKSEA_MULTIYEAR_PHY_007_004/cmems_mod_blk_phy-cur_my_2.5km_P1D-m/*
    # -O /mnt/miyuan/AI4Physics/Data/bs/data.zip --ftp-user=ymi --ftp-password=My12345678 -r
    _download_raw_data()
