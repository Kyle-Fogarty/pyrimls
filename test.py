import sys
sys.path.append('src/pyrimls')


import torch 
import src.pyrimls as pyr
from tqdm import tqdm

reconstructor = pyr.RobustImplicitMLS_batched()

def potential(batch, points, normals):
    return torch.vmap(reconstructor.potential)(batch, points, normals)

# Generate a sphere point cloud
def generate_sphere_points(n_points=1000, radius=0.5, noise=0.0):
    # Generate random points on a sphere
    theta = torch.rand(n_points) * 2 * torch.pi
    phi = torch.acos(2 * torch.rand(n_points) - 1)
    
    # Convert to Cartesian coordinates
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)
    
    points = torch.stack([x, y, z], dim=1)
    
    # Calculate normals (for a sphere, normals point outward from center)
    normals = points / radius
    
    # Add some noise if specified
    if noise > 0:
        points += torch.randn_like(points) * noise
        normals = torch.nn.functional.normalize(normals, dim=1)
    
    return points, normals

# Generate test data
points, normals = generate_sphere_points(n_points=2000, radius=0.5, noise=0.01)


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
B = 32
values = []
for i in tqdm(range(0, len(grid_points), batch_size)):
    batch = grid_points[i:i+batch_size]
    # print(batch.shape, points.shape, normals.shape)
    # batch = batch
    # points = points.repeat(5, 1)
    # normals = normals.repeat(5, 1)
    potential = reconstructor.potential(batch.cuda().unsqueeze(0).repeat(B, 1, 1), points.cuda().unsqueeze(0).repeat(B, 1, 1), normals.cuda().unsqueeze(0).repeat(B, 1, 1))
    # print(potential.shape)
    values.append(potential)

sdf = torch.hstack(values)
print(sdf.shape)
sdf = sdf.reshape(sdf.shape[0], N, N, N)


print(sdf)