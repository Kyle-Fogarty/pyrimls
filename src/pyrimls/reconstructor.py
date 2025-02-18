import torch
import torch.nn.functional as F

class RIMLS:
    def __init__(self, points, normals):
        """
        Initialize RIMLS with mesh data
        
        Args:
            mesh_vertices: (N, 3) tensor of vertex positions
            mesh_normals: (N, 3) tensor of vertex normals
        """
        self.vertices = points
        self.normals  = normals
        
        # Parameters
        self.sigma_n = 0.8
        self.sigma_r = 0.0
        self.refitting_threshold = 1e-3
        self.min_refitting_iters = 1
        self.max_refitting_iters = 3
        self.projection_accuracy = 0.0001
        self.max_projection_iters = 15
        
        # Compute average point spacing for projection epsilon
        with torch.no_grad():
            diffs = self.vertices.unsqueeze(1) - self.vertices.unsqueeze(0)
            distances = torch.norm(diffs, dim=2)
            distances.diagonal().fill_(float('inf'))
            self.average_spacing = distances.min(dim=1)[0].mean()

    def compute_weights_and_derivatives(self, query_points, neighborhood_indices, h):
        """
        Compute weights and their derivatives for the given query points
        
        Args:
            query_points: (Q, 3) tensor of query positions
            neighborhood_indices: (Q, K) tensor of indices for K nearest neighbors
            h: (Q,) tensor of bandwidth parameters
            
        Returns:
            weights: (Q, K) tensor of weights
            weight_gradients: (Q, K, 3) tensor of weight gradients
            weight_derivatives: (Q, K) tensor of weight derivatives
        """
        # Get positions of neighbors
        neighbors = self.vertices[neighborhood_indices]  # (Q, K, 3)
        
        # Compute differences and distances
        diffs = query_points.unsqueeze(1) - neighbors  # (Q, K, 3)
        distances = torch.norm(diffs, dim=2)  # (Q, K)
        
        # Reshape h to match the dimensions for broadcasting
        h = h.unsqueeze(1).expand(-1, neighborhood_indices.shape[1])  # (Q, K)
        weights = torch.exp(-distances**2 / (h**2))  # (Q, K)
        
        # Compute derivatives
        weight_gradients = -2 * weights.unsqueeze(-1) * diffs / (h**2).unsqueeze(-1)  # (Q, K, 3)
        weight_derivatives = -2 * weights / (h**2)  # (Q, K)
        
        return weights, weight_gradients, weight_derivatives

    def compute_potential_and_gradient(self, x, neighborhood_indices, h):
        """
        Compute the potential and gradient at query points
        
        Args:
            x: (Q, 3) tensor of query positions
            neighborhood_indices: (Q, K) tensor of indices for K nearest neighbors
            h: bandwidth parameter
            
        Returns:
            potential: (Q,) tensor of potential values
            gradient: (Q, 3) tensor of gradients
        """
        device    = x.device
        n_queries = x.shape[0]
        n_neighbors = neighborhood_indices.shape[1]
        
        # Initialize cached values
        cached_weights = torch.zeros(n_queries, n_neighbors, device=device)
        cached_refitting_weights = torch.ones(n_queries, n_neighbors, device=device)
        
        # Get neighbor positions and normals
        neighbors        = self.vertices[neighborhood_indices]  # (Q, K, 3)
        neighbor_normals = self.normals[neighborhood_indices]  # (Q, K, 3)
        
        # Compute weights and derivatives
        weights, weight_gradients, _ = self.compute_weights_and_derivatives(x, neighborhood_indices, h)
        cached_weights = weights
        
        # RIMLS iteration
        grad = torch.zeros_like(x)
        for iter_count in range(self.max_refitting_iters):
            prev_grad = grad.clone()
            
            if iter_count > 0:
                # Compute refitting weights
                normal_diff = neighbor_normals - prev_grad.unsqueeze(1)
                refitting_weights = torch.exp(-torch.sum(normal_diff**2, dim=2) / self.sigma_n**2)
                cached_refitting_weights = refitting_weights
            
            # Compute total weights
            total_weights = cached_weights * cached_refitting_weights  # (Q, K)
            
            # Compute differences and dot products
            diffs = x.unsqueeze(1) - neighbors  # (Q, K, 3)
            f = torch.sum(diffs * neighbor_normals, dim=2)  # (Q, K)
            
            # Compute weighted sums
            sum_w = total_weights.sum(dim=1)  # (Q,)
            sum_wf = (total_weights * f).sum(dim=1)  # (Q,)
            
            # Compute potential and gradient
            potential = sum_wf / sum_w  # (Q,)
            
            weighted_gradients = weight_gradients * cached_refitting_weights.unsqueeze(-1)  # (Q, K, 3)
            sum_grad_w = weighted_gradients.sum(dim=1)  # (Q, 3)
            sum_grad_wf = (weighted_gradients * f.unsqueeze(-1)).sum(dim=1)  # (Q, 3)
            sum_wn = (total_weights.unsqueeze(-1) * neighbor_normals).sum(dim=1)  # (Q, 3)
            
            grad = (-sum_grad_w * potential.unsqueeze(-1) + sum_grad_wf + sum_wn) / sum_w.unsqueeze(-1)
            
            # Check convergence
            if iter_count >= self.min_refitting_iters:
                grad_diff = torch.norm(grad - prev_grad, dim=1)
                if torch.all(grad_diff <= self.refitting_threshold):
                    break
                    
        return potential, grad

    def potential(self, points):
        """
        Compute the potential value at query points
        
        Args:
            points: (N, 3) tensor of query points
            
        Returns:
            potential: (N,) tensor of potential values
        """
        device = points.device
        
        # Find K nearest neighbors for each point
        # with torch.no_grad():
        diffs = points.unsqueeze(1) - self.vertices.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)
        k = min(20, self.vertices.shape[0])
        _, neighborhood_indices = torch.topk(distances, k, dim=1, largest=False)
        
        # Compute local feature size (average distance to neighbors)
        neighbor_distances = distances.gather(1, neighborhood_indices)
        h = neighbor_distances.mean(dim=1)  # (N,)
    
        # Compute potential
        potential, _ = self.compute_potential_and_gradient(points, neighborhood_indices, h)
        return potential

    def project(self, points):
        """
        Project points onto the surface
        
        Args:
            points: (N, 3) tensor of points to project
            
        Returns:
            projected_points: (N, 3) tensor of projected points
            normals: (N, 3) tensor of surface normals at projected points
        """
        device = points.device
        projected = points.clone()
        normals = torch.zeros_like(points)
        
        # Find K nearest neighbors for each point
        with torch.no_grad():
            diffs = points.unsqueeze(1) - self.vertices.unsqueeze(0)
            distances = torch.norm(diffs, dim=2)
            k = min(20, self.vertices.shape[0])
            _, neighborhood_indices = torch.topk(distances, k, dim=1, largest=False)
        
        # Compute local feature size (average distance to neighbors)
        h = distances.gather(1, neighborhood_indices).mean(dim=1)
        
        epsilon = self.average_spacing * self.projection_accuracy
        
        # Projection iterations
        for _ in range(self.max_projection_iters):
            potential, gradient = self.compute_potential_and_gradient(
                projected, neighborhood_indices, h)
            
            # Normalize gradient to get normal
            normal = F.normalize(gradient, dim=1)
            normals = normal
            
            # Update position
            delta = potential.unsqueeze(-1) * normal
            projected = projected - delta
            
            # Check convergence
            if torch.all(torch.abs(potential) <= epsilon):
                break
                
        return projected, normals
    





import torch
import torch.nn.functional as F

class RIMLS_Functional:
    def __init__(self):
        """
        Initialize RIMLS with default parameters
        """
        # Parameters
        self.sigma_n = 0.8
        self.sigma_r = 0.0
        self.refitting_threshold = 1e-3
        self.min_refitting_iters = 1
        self.max_refitting_iters = 3
        self.projection_accuracy = 0.0001
        self.max_projection_iters = 15

    def _compute_average_spacing(self, vertices):
        """
        Compute average point spacing for projection epsilon
        
        Args:
            vertices: (N, 3) tensor of vertex positions
            
        Returns:
            float: average spacing between points
        """
        with torch.no_grad():
            diffs = vertices.unsqueeze(1) - vertices.unsqueeze(0)
            distances = torch.norm(diffs, dim=2)
            distances.diagonal().fill_(float('inf'))
            return distances.min(dim=1)[0].mean()

    def compute_weights_and_derivatives(self, query_points, vertices, neighborhood_indices, h):
        """
        Compute weights and their derivatives for the given query points
        
        Args:
            query_points: (Q, 3) tensor of query positions
            vertices: (N, 3) tensor of vertex positions
            neighborhood_indices: (Q, K) tensor of indices for K nearest neighbors
            h: (Q,) tensor of bandwidth parameters
            
        Returns:
            weights: (Q, K) tensor of weights
            weight_gradients: (Q, K, 3) tensor of weight gradients
            weight_derivatives: (Q, K) tensor of weight derivatives
        """
        # Get positions of neighbors
        neighbors = vertices[neighborhood_indices]  # (Q, K, 3)
        
        # Compute differences and distances
        diffs = query_points.unsqueeze(1) - neighbors  # (Q, K, 3)
        distances = torch.norm(diffs, dim=2)  # (Q, K)
        
        # Reshape h to match the dimensions for broadcasting
        h = h.unsqueeze(1).expand(-1, neighborhood_indices.shape[1])  # (Q, K)
        weights = torch.exp(-distances**2 / (h**2))  # (Q, K)
        
        # Compute derivatives
        weight_gradients = -2 * weights.unsqueeze(-1) * diffs / (h**2).unsqueeze(-1)  # (Q, K, 3)
        weight_derivatives = -2 * weights / (h**2)  # (Q, K)
        
        return weights, weight_gradients, weight_derivatives

    def compute_potential_and_gradient(self, x, vertices, normals, neighborhood_indices, h):
        """
        Compute the potential and gradient at query points
        
        Args:
            x: (Q, 3) tensor of query positions
            vertices: (N, 3) tensor of vertex positions
            normals: (N, 3) tensor of vertex normals
            neighborhood_indices: (Q, K) tensor of indices for K nearest neighbors
            h: bandwidth parameter
            
        Returns:
            potential: (Q,) tensor of potential values
            gradient: (Q, 3) tensor of gradients
        """
        device = x.device
        n_queries = x.shape[0]
        n_neighbors = neighborhood_indices.shape[1]
        
        # Initialize cached values
        cached_weights = torch.zeros(n_queries, n_neighbors, device=device)
        cached_refitting_weights = torch.ones(n_queries, n_neighbors, device=device)
        
        # Get neighbor positions and normals
        neighbors = vertices[neighborhood_indices]  # (Q, K, 3)
        neighbor_normals = normals[neighborhood_indices]  # (Q, K, 3)
        
        # Compute weights and derivatives
        weights, weight_gradients, _ = self.compute_weights_and_derivatives(x, vertices, neighborhood_indices, h)
        cached_weights = weights
        
        # RIMLS iteration
        grad = torch.zeros_like(x)
        for iter_count in range(self.max_refitting_iters):
            prev_grad = grad.clone()
            
            if iter_count > 0:
                # Compute refitting weights
                normal_diff = neighbor_normals - prev_grad.unsqueeze(1)
                refitting_weights = torch.exp(-torch.sum(normal_diff**2, dim=2) / self.sigma_n**2)
                cached_refitting_weights = refitting_weights
            
            # Compute total weights
            total_weights = cached_weights * cached_refitting_weights  # (Q, K)
            
            # Compute differences and dot products
            diffs = x.unsqueeze(1) - neighbors  # (Q, K, 3)
            f = torch.sum(diffs * neighbor_normals, dim=2)  # (Q, K)
            
            # Compute weighted sums
            sum_w = total_weights.sum(dim=1)  # (Q,)
            sum_wf = (total_weights * f).sum(dim=1)  # (Q,)
            
            # Compute potential and gradient
            potential = sum_wf / sum_w  # (Q,)
            
            weighted_gradients = weight_gradients * cached_refitting_weights.unsqueeze(-1)  # (Q, K, 3)
            sum_grad_w = weighted_gradients.sum(dim=1)  # (Q, 3)
            sum_grad_wf = (weighted_gradients * f.unsqueeze(-1)).sum(dim=1)  # (Q, 3)
            sum_wn = (total_weights.unsqueeze(-1) * neighbor_normals).sum(dim=1)  # (Q, 3)
            
            grad = (-sum_grad_w * potential.unsqueeze(-1) + sum_grad_wf + sum_wn) / sum_w.unsqueeze(-1)
            
            # Check convergence
            if iter_count >= self.min_refitting_iters:
                grad_diff = torch.norm(grad - prev_grad, dim=1)
                if torch.all(grad_diff <= self.refitting_threshold):
                    break
                    
        return potential, grad

    def potential(self, points, vertices, normals):
        """
        Compute the potential value at query points
        
        Args:
            points: (N, 3) tensor of query points
            vertices: (M, 3) tensor of vertex positions
            normals: (M, 3) tensor of vertex normals
            
        Returns:
            potential: (N,) tensor of potential values
        """
        device = points.device
        
        # Find K nearest neighbors for each point
        diffs = points.unsqueeze(1) - vertices.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)
        k = min(20, vertices.shape[0])
        _, neighborhood_indices = torch.topk(distances, k, dim=1, largest=False)
        
        # Compute local feature size (average distance to neighbors)
        neighbor_distances = distances.gather(1, neighborhood_indices)
        h = neighbor_distances.mean(dim=1)  # (N,)
    
        # Compute potential
        potential, _ = self.compute_potential_and_gradient(points, vertices, normals, neighborhood_indices, h)
        return potential

    def project(self, points, vertices, normals):
        """
        Project points onto the surface
        
        Args:
            points: (N, 3) tensor of points to project
            vertices: (M, 3) tensor of vertex positions
            normals: (M, 3) tensor of vertex normals
            
        Returns:
            projected_points: (N, 3) tensor of projected points
            normals: (N, 3) tensor of surface normals at projected points
        """
        device = points.device
        projected = points.clone()
        output_normals = torch.zeros_like(points)
        
        # Find K nearest neighbors for each point
        with torch.no_grad():
            diffs = points.unsqueeze(1) - vertices.unsqueeze(0)
            distances = torch.norm(diffs, dim=2)
            k = min(20, vertices.shape[0])
            _, neighborhood_indices = torch.topk(distances, k, dim=1, largest=False)
        
        # Compute local feature size (average distance to neighbors)
        h = distances.gather(1, neighborhood_indices).mean(dim=1)
        
        # Compute average spacing for projection epsilon
        average_spacing = self._compute_average_spacing(vertices)
        epsilon = average_spacing * self.projection_accuracy
        
        # Projection iterations
        for _ in range(self.max_projection_iters):
            potential, gradient = self.compute_potential_and_gradient(
                projected, vertices, normals, neighborhood_indices, h)
            
            # Normalize gradient to get normal
            normal = F.normalize(gradient, dim=1)
            output_normals = normal
            
            # Update position
            delta = potential.unsqueeze(-1) * normal
            projected = projected - delta
            
            # Check convergence
            if torch.all(torch.abs(potential) <= epsilon):
                break
                
        return projected, output_normals