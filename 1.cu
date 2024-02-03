#include "TriangleMeshDistance.h"
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

typedef tmd::TriangleMeshDistance<thrust::host_vector, true> HostTBuild;
typedef tmd::TriangleMeshDistance<thrust::host_vector, false> HostTQuery;
typedef tmd::TriangleMeshDistance<thrust::device_vector, false> DeviTQuery;

#define safe_cuda(CODE)                                                        \
  {                                                                            \
    cudaError_t err = CODE;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cout << "CUDA error:" << cudaGetErrorString(err) << std::endl;      \
    }                                                                          \
  }

__global__ void v(void *t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  ((char *)t)[i] = i;
}

double point_AABB_signed(const tmd::Vec3d& point, const tmd::Vec3d& bottom, const tmd::Vec3d& top)
{
	const tmd::Vec3d dx = { std::max(bottom[0] - point[0], point[0] - top[0]),
									std::max(bottom[1] - point[1], point[1] - top[1]),
									std::max(bottom[2] - point[2], point[2] - top[2]) };

	const double max_dx = std::max(dx[0], std::max(dx[1], dx[2]));
	if (max_dx < 0.0) { // Inside
		return max_dx;
	}
	else { // Outside
		double dist_sq = 0.0;
		for (int i = 0; i < 3; i++) {
			if (dx[i] > 0.0) {
				dist_sq += dx[i] * dx[i];
			}

		}
		return std::sqrt(dist_sq);
	}
}

__global__ void vvv(DeviTQuery & t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // t.signed_distance({1,1,1});
}

int main() {
  void *dst;
  size_t size = 100;
  safe_cuda(cudaMalloc(&dst, size));
  std::vector<char> s(100);
  safe_cuda(cudaMemcpy(dst, s.data(), size, cudaMemcpyHostToDevice));
  v<<<dim3(10, 1, 1), dim3(10, 1, 1)>>>(dst);
  safe_cuda(cudaDeviceSynchronize());
  safe_cuda(cudaMemcpy(s.data(), dst, size, cudaMemcpyDeviceToHost));
  for (auto &vv : s) {
    std::cout << int(vv) << std::endl;
  }
  safe_cuda(cudaFree(dst));

  thrust::host_vector<char> ss(100);
  thrust::device_vector<char> d_vec = ss;
  v<<<dim3(10, 1, 1), dim3(10, 1, 1)>>>((void *)(d_vec.data().get()));
  safe_cuda(cudaDeviceSynchronize());
  ss = d_vec;
  for (auto &vv : ss) {
    std::cout << int(vv) << std::endl;
  }

    std::vector<tmd::Vec3d> vertices = { { 1, -1, -1 }, { 1, 0, -1 }, { 1, 1, -1 }, { 1, -1, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 1, -1, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { -1, -1, -1 }, { -1, 0, -1 }, { -1, 1, -1 }, { -1, -1, 0 }, { -1, 0, 0 }, { -1, 1, 0 }, { -1, -1, 1 }, { -1, 0, 1 }, { -1, 1, 1 }, { 0, 1, -1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 0, -1, -1 }, { 0, -1, 0 }, { 0, -1, 1 }, { 0, 0, 1 }, { 0, 0, -1 } };
		std::vector<std::array<int, 3>> connectivity = { { 0, 1, 3 }, { 1, 4, 3 }, { 1, 2, 4 }, { 2, 5, 4 }, { 3, 4, 6 }, { 4, 7, 6 }, { 4, 5, 7 }, { 5, 8, 7 }, { 12, 10, 9 }, { 12, 13, 10 }, { 13, 11, 10 }, { 13, 14, 11 }, { 15, 13, 12 }, { 15, 16, 13 }, { 16, 14, 13 }, { 16, 17, 14 }, { 14, 18, 11 }, { 14, 19, 18 }, { 19, 2, 18 }, { 19, 5, 2 }, { 17, 19, 14 }, { 17, 20, 19 }, { 20, 5, 19 }, { 20, 8, 5 }, { 9, 21, 12 }, { 21, 22, 12 }, { 21, 0, 22 }, { 0, 3, 22 }, { 12, 22, 15 }, { 22, 23, 15 }, { 22, 3, 23 }, { 3, 6, 23 }, { 15, 23, 16 }, { 23, 24, 16 }, { 23, 6, 24 }, { 6, 7, 24 }, { 16, 24, 17 }, { 24, 20, 17 }, { 24, 7, 20 }, { 7, 8, 20 }, { 10, 21, 9 }, { 10, 25, 21 }, { 25, 0, 21 }, { 25, 1, 0 }, { 11, 25, 10 }, { 11, 18, 25 }, { 18, 1, 25 }, { 18, 2, 1 } };

		HostTBuild mesh_distance;
    mesh_distance.construct(vertices, connectivity);
    HostTQuery mesh_distance_;
    DeviTQuery mesh_distance_Device;
    
    mesh_distance_Device.vertices                = mesh_distance.vertices; 
		mesh_distance_Device.triangles               = mesh_distance.triangles; 
		mesh_distance_Device.nodes                   = mesh_distance.nodes; 
		mesh_distance_Device.pseudonormals_triangles = mesh_distance.pseudonormals_triangles; 
		mesh_distance_Device.pseudonormals_edges     = mesh_distance.pseudonormals_edges; 
		mesh_distance_Device.pseudonormals_vertices  = mesh_distance.pseudonormals_vertices; 
		mesh_distance_Device.root_bv                 = mesh_distance.root_bv;
		mesh_distance_Device.is_constructed          = mesh_distance.is_constructed; 

    mesh_distance_.vertices                = mesh_distance.vertices; 
		mesh_distance_.triangles               = mesh_distance.triangles; 
		mesh_distance_.nodes                   = mesh_distance.nodes; 
		mesh_distance_.pseudonormals_triangles = mesh_distance.pseudonormals_triangles; 
		mesh_distance_.pseudonormals_edges     = mesh_distance.pseudonormals_edges; 
		mesh_distance_.pseudonormals_vertices  = mesh_distance.pseudonormals_vertices; 
		mesh_distance_.root_bv                 = mesh_distance.root_bv;
		mesh_distance_.is_constructed          = mesh_distance.is_constructed; 


    for (float x = -2; x < 2; x += 0.13) {
			for (float y = -2; y < 2; y += 0.13) {
				for (float z = -2; z < 2; z += 0.13) {
					const auto result = mesh_distance_.signed_distance({ x, y, z });
					const float exact = point_AABB_signed({ x, y, z }, { -1, -1, -1 }, { 1, 1, 1 });
					std::cout << (std::abs(result.distance - exact) < 1e-5) << std::endl;
				}
			}
		}
    vvv<<<dim3(10, 1, 1), dim3(10, 1, 1)>>>(mesh_distance_Device);
}