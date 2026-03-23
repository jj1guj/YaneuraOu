#include "unpack.cuh"

namespace YaneuraOu {

// 旧実装は FType=int として int* にビット操作した値を書き込み、
// TRT が float* として読む型エイリアシング違反があった。
// CUDA 13.1 以降のコンパイラ最適化でこれが壊れる (常にゼロが出力される) ため、
// DType (float / __half) に直接キャストする安全な実装に変更。
// 符号付き char のシフトも unsigned char に変更して実装定義動作を排除した。

constexpr int features1_size = 62;
constexpr int features2_size = 57;

__global__ void unpack_features1_kernel(const unsigned char* __restrict__ p1, DType* __restrict__ x1, int max_tid) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= max_tid) return;
	int x1_offset = tid * 81;
	for (int i = 0; i < 81; ++i) {
		int j = x1_offset + i;
		x1[j] = (DType)((p1[j >> 3] >> (j & 7)) & 1);
	}
}

__global__ void unpack_features2_kernel(const unsigned char* __restrict__ p2, DType* __restrict__ x2, int max_tid) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= max_tid) return;
	int x2_offset = tid * 81;
	DType v = (DType)((p2[tid >> 3] >> (tid & 7)) & 1);
	for (int i = 0; i < 81; ++i) {
		x2[x2_offset + i] = v;
	}
}

void unpack_features1(const int batch_size, PType* p1, DType* x1, cudaStream_t stream)
{
	unpack_features1_kernel<<<batch_size, features1_size, 0, stream>>>((const unsigned char*)p1, x1, batch_size * features1_size);
}

void unpack_features2(const int batch_size, PType* p2, DType* x2, cudaStream_t stream)
{
	unpack_features2_kernel<<<batch_size, features2_size, 0, stream>>>((const unsigned char*)p2, x2, batch_size * features2_size);
}

} // namespace YaneuraOu
