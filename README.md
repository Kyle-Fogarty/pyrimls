


```python
import torch 
import src.pyrimls as pyr
import diso

reconstructor = pyr.RobustImplicitMLS()

differentiable_marching_cubes  = DiffMC(dtype=torch.float32)

# (1) Setup the pointcloud data
points, normals = torch.from_numpy(points).float(), torch.from_numpy(normals).float()


# (2) Setup the MC grid
min_plc, max_plc = -1, 1 # Bounds of the marching cube box.
N = 100                  # march cubes resolution


grid_points = torch.stack(torch.meshgrid(torch.linspace(min_plc, max_plc, N),
                                         torch.linspace(min_plc, max_plc, N),
                                         torch.linspace(min_plc, max_plc, N)),
                                         dim=-1)


grid_points = grid_points.reshape(-1, 3)

# (3) Compute the potential/SDF values
batch_size = 10000

values = []
for i in tqdm(range(0, len(grid_points), batch_size)):
    batch = grid_points[i:i+batch_size]
    potential = reconstructor.potential(batch, plc, normals)
    values.append(potential)

sdf = torch.cat(values)
sdf = sdf.reshape(N, N, N)


# (4) Marching cubes
verts, faces = differentiable_marching_cubes(sdf, None, isovalue=0)
```