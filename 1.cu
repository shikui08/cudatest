#include "TriangleMeshDistance.h"
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

typedef tmd::TriangleMeshDistance<thrust::host_vector> HostT;
typedef tmd::TriangleMeshDistance<thrust::device_vector> DeviT;

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

		HostT mesh_distance(vertices, connectivity);
        DeviT mesh_distance_Device;
        
        mesh_distance_Device.vertices                = mesh_distance.vertices; 
		mesh_distance_Device.triangles               = mesh_distance.triangles; 
		mesh_distance_Device.nodes                   = mesh_distance.nodes; 
		mesh_distance_Device.pseudonormals_triangles = mesh_distance.pseudonormals_triangles; 
		mesh_distance_Device.pseudonormals_edges     = mesh_distance.pseudonormals_edges; 
		mesh_distance_Device.pseudonormals_vertices  = mesh_distance.pseudonormals_vertices; 
		mesh_distance_Device.root_bv                 = mesh_distance.root_bv;
		mesh_distance_Device.is_constructed          = mesh_distance.is_constructed; 
}