// Correctness harness for the standalone compute kernels (kernels/).
// All kernels are driven at GPT-124M training shapes (B=16, seq=1024, C=768,
// vocab=50304, 12 heads x 64) so the tests exercise real transformer tensor sizes.
// Host random init -> CPU fp32 reference -> kernel launch -> max-absolute-error vs tolerance.
// unified_reduce_standalone is excluded (it has its own main() self-test).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while(0)

static int g_fail = 0;

// max absolute error between host vector and device buffer.
static double max_abs_err(const std::vector<float>& ref, const float* d_out, size_t n) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    double e = 0;
    for (size_t i = 0; i < n; ++i) e = std::max(e, (double)std::abs(h[i] - ref[i]));
    return e;
}

static void report(const std::string& name, double err, double tol, const char* cuda_err) {
    bool pass = (cuda_err == nullptr) && (err <= tol);
    if (!pass) g_fail++;
    printf("%-40s max_abs_err=%.3e tol=%.0e  %s\n", name.c_str(), err, tol, pass ? "PASS" : "FAIL");
    if (cuda_err) printf("    ^^^ CUDA error: %s\n", cuda_err);
}

// Capture launch + sync error after a kernel; returns nullptr if clean.
static const char* sync_err() {
    cudaError_t le = cudaGetLastError();
    cudaError_t se = cudaDeviceSynchronize();
    if (le != cudaSuccess) return cudaGetErrorString(le);
    if (se != cudaSuccess) return cudaGetErrorString(se);
    return nullptr;
}

static std::vector<float> rand_vec(size_t n, uint64_t seed, float lo = -1.f, float hi = 1.f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// peak |value| of a reference vector, used to scale relative tolerances.
static double vmax_abs(const std::vector<float>& v) {
    double m = 0; for (float x : v) m = std::max(m, (double)std::fabs(x));
    return m;
}

// ---------------------------------------------------------------------------
// CPU FlashAttention reference (fp32). Tensors are [B, H, T, Hd] contiguous,
// LSE/D are [B, H, T]. These mirror exactly what the TF32 tensor-core kernels
// compute: forward = softmax(scale * Q@K^T) @ V with LSE = max + log(sumexp);
// backward = the standard FlashAttention dQ/dK/dV using D = rowsum(dO * O).
// ---------------------------------------------------------------------------
static void attn_ref_forward(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    int B, int H, int T, int Hd, float scale, bool causal,
    std::vector<float>& O, std::vector<float>& LSE)
{
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
        const int64_t base = ((int64_t)b * H + h) * T * Hd;
        const int64_t lbase = ((int64_t)b * H + h) * T;
        for (int i = 0; i < T; ++i) {
            const float* qi = &Q[base + (int64_t)i * Hd];
            const int jmax = causal ? i : (T - 1);
            float m = -1e30f;
            for (int j = 0; j <= jmax; ++j) {
                const float* kj = &K[base + (int64_t)j * Hd];
                double d = 0; for (int k = 0; k < Hd; ++k) d += (double)qi[k] * kj[k];
                float s = (float)(d * scale); if (s > m) m = s;
            }
            double l = 0, oacc[64] = {0};   // Hd <= 64 in this harness
            for (int j = 0; j <= jmax; ++j) {
                const float* kj = &K[base + (int64_t)j * Hd];
                const float* vj = &V[base + (int64_t)j * Hd];
                double d = 0; for (int k = 0; k < Hd; ++k) d += (double)qi[k] * kj[k];
                float e = std::exp((float)(d * scale) - m); l += e;
                for (int k = 0; k < Hd; ++k) oacc[k] += (double)e * vj[k];
            }
            double inv = (l > 0) ? 1.0 / l : 0.0;
            for (int k = 0; k < Hd; ++k) O[base + (int64_t)i * Hd + k] = (float)(oacc[k] * inv);
            LSE[lbase + i] = (float)(m + std::log(l));
        }
    }
}

// Fills D (rowsum(dO*O)) and dQ/dK/dV. O and LSE come from attn_ref_forward.
static void attn_ref_backward(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    const std::vector<float>& O, const std::vector<float>& dO, const std::vector<float>& LSE,
    int B, int H, int T, int Hd, float scale, bool causal,
    std::vector<float>& D, std::vector<float>& dQ, std::vector<float>& dK, std::vector<float>& dV)
{
    std::fill(dQ.begin(), dQ.end(), 0.f);
    std::fill(dK.begin(), dK.end(), 0.f);
    std::fill(dV.begin(), dV.end(), 0.f);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
        const int64_t base = ((int64_t)b * H + h) * T * Hd;
        const int64_t lbase = ((int64_t)b * H + h) * T;
        for (int i = 0; i < T; ++i) {
            const float* qi  = &Q[base + (int64_t)i * Hd];
            const float* doi = &dO[base + (int64_t)i * Hd];
            const float* oi  = &O[base + (int64_t)i * Hd];
            double Di = 0; for (int k = 0; k < Hd; ++k) Di += (double)doi[k] * oi[k];
            D[lbase + i] = (float)Di;
            const float Li = LSE[lbase + i];
            const int jmax = causal ? i : (T - 1);
            float* dqi = &dQ[base + (int64_t)i * Hd];
            for (int j = 0; j <= jmax; ++j) {
                const float* kj = &K[base + (int64_t)j * Hd];
                const float* vj = &V[base + (int64_t)j * Hd];
                float* dkj = &dK[base + (int64_t)j * Hd];
                float* dvj = &dV[base + (int64_t)j * Hd];
                double sd = 0, dp = 0;
                for (int k = 0; k < Hd; ++k) { sd += (double)qi[k] * kj[k]; dp += (double)doi[k] * vj[k]; }
                float p  = std::exp((float)(sd * scale) - Li);
                float ds = (float)(p * (dp - Di) * scale);
                for (int k = 0; k < Hd; ++k) {
                    dqi[k] += ds * kj[k];
                    dkj[k] += ds * qi[k];
                    dvj[k] += p  * doi[k];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel forward declarations (must match the linked objects, incl. namespaces).
// ---------------------------------------------------------------------------

// global scope
template<typename T> __global__ void gelu_forward_sm89_kernel(const T*, T*, int64_t);
template<typename T, typename AccT> __global__ void layer_norm_forward_sm89_kernel(
    const T*, const T*, const T*, T*, AccT*, AccT*, int, AccT);
template<typename T> __global__ void ln_backward_gamma_beta_sm89_kernel(
    const T*, const T*, const float*, const float*, float*, float*, int, int);
template<typename T, typename T_idx> __global__ void sparse_ce_forward_kernel_vec_save_stats(
    const T*, const T_idx*, T*, float*, float*, int64_t, int64_t);
template<typename T, typename T_idx> __global__ void sparseCENormalize_from_stats(
    const T*, const T_idx*, T*, const float*, const float*, int64_t, int64_t, const T*, float);

// OwnTensor::cuda
namespace OwnTensor { namespace cuda {
template<typename T> __global__ void gelu_backward_sm89_kernel(const T*, const T*, T*, int64_t);
template<typename T> __global__ void ln_backward_input_sm89_kernel(
    const T*, const T*, const float*, const float*, const T*, T*, int);
}}

// multi_tensor_adam: by-value metadata struct, global scope.
static const int MTA_CHUNK = 32768, MTA_MAXB = 320, MTA_MAXT = 48;
struct AdamLaunchMetadata {
    float* params[MTA_MAXT]; float* grads[MTA_MAXT];
    float* ms[MTA_MAXT];     float* vs[MTA_MAXT];
    int64_t sizes[MTA_MAXT];
    unsigned char block_to_tensor[MTA_MAXB];
    int block_to_chunk[MTA_MAXB];
};
__global__ void multi_tensor_adam_sm89_kernel(AdamLaunchMetadata, float, float, float, float, float, float, float);

// precompute_D: global-scope struct + kernel.
struct MemEfficientBwdParams {
    const float* Q; const float* K; const float* V; const float* O; const float* dO; const float* LSE;
    float* D; float* dQ; float* dK; float* dV;
    int B, nh, T; float scale; bool is_causal;
    int64_t q_strideB, q_strideM, q_strideH;
    int64_t k_strideB, k_strideM, k_strideH;
    int64_t v_strideB, v_strideM, v_strideH;
    int64_t o_strideB, o_strideM, o_strideH;
    int64_t do_strideB, do_strideM, do_strideH;
    int64_t dq_strideB, dq_strideM, dq_strideH;
    int64_t dk_strideB, dk_strideM, dk_strideH;
    int64_t dv_strideB, dv_strideM, dv_strideH;
    int64_t lse_strideB, lse_strideH;
    int64_t d_strideB, d_strideH;
};
template<int HeadDim> __global__ void mem_efficient_bwd_precompute_D_sm89(MemEfficientBwdParams);

// fused_attn_forward: by-value param struct, global scope (matches the kernel's header).
struct MemEfficientFwdParams {
    const float* Q; const float* K; const float* V; float* O; float* LSE;
    int B; int nh; int64_t T;
    float scale; bool is_causal; float dropout_p; const float* dropout_mask;
    int64_t q_strideB, q_strideM, q_strideH;
    int64_t k_strideB, k_strideM, k_strideH;
    int64_t v_strideB, v_strideM, v_strideH;
    int64_t o_strideB, o_strideM, o_strideH;
    int64_t lse_strideB, lse_strideH;
};
template<int HeadDim, int BQ_TILE, int BK_TILE, int MaxBlocksPerSM>
__global__ void fused_attn_forward_kernel_tc_sm89(MemEfficientFwdParams);

// mem_efficient_bwd_unified: lives in namespace OwnTensor with its own param struct
// (same layout as the global precompute_D struct, but the namespace matters for mangling).
namespace OwnTensor {
struct MemEfficientBwdParams {
    const float* Q; const float* K; const float* V; const float* O; const float* dO; const float* LSE;
    float* D; float* dQ; float* dK; float* dV;
    int B; int nh; int T; float scale; bool is_causal;
    int64_t q_strideB, q_strideM, q_strideH;
    int64_t k_strideB, k_strideM, k_strideH;
    int64_t v_strideB, v_strideM, v_strideH;
    int64_t o_strideB, o_strideM, o_strideH;
    int64_t do_strideB, do_strideM, do_strideH;
    int64_t dq_strideB, dq_strideM, dq_strideH;
    int64_t dk_strideB, dk_strideM, dk_strideH;
    int64_t dv_strideB, dv_strideM, dv_strideH;
    int64_t lse_strideB, lse_strideH;
    int64_t d_strideB, d_strideH;
};
template<int HeadDim, bool Causal>
__global__ void mem_efficient_bwd_unified_kernel_exp12(MemEfficientBwdParams);
}

int main() {
    int dev = 0; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    const double TOL_EW = 5e-2;   // elementwise / reduction
    const double TOL_TF = 2e-2;   // TF32 tensor-core paths

    // ---- GPT-124M training shapes (b=16, seq=1024, c=768, vocab=50304, 12 heads x 64) ----
    // These drive every kernel below so the harness exercises real transformer tensor
    // sizes rather than toy dimensions.
    const int   GPT_B     = 16;
    const int   GPT_T     = 1024;
    const int   GPT_C     = 768;
    const int   GPT_4C    = 4 * GPT_C;      // 3072, MLP hidden
    const int   GPT_VOCAB = 50304;
    const int   GPT_HEADS = 12;
    const int   GPT_HD    = 64;             // head dim
    const int64_t GPT_ROWS = (int64_t)GPT_B * GPT_T;   // 16384, tokens = LN/CE rows
    printf("GPT-124M shapes: B=%d T=%d C=%d 4C=%d vocab=%d heads=%d head_dim=%d (rows=B*T=%lld)\n\n",
           GPT_B, GPT_T, GPT_C, GPT_4C, GPT_VOCAB, GPT_HEADS, GPT_HD, (long long)GPT_ROWS);

    auto tanh_gelu = [](float x) {
        const float c = 0.7978845608028654f, k = 0.044715f;
        return 0.5f * x * (1.0f + std::tanh(c * (x + k * x * x * x)));
    };

    // ---- gelu_forward (float) ---- GPT MLP hidden activation: rows * 4C elements.
    {
        int64_t n = GPT_ROWS * GPT_4C; auto hx = rand_vec(n, 1, -4.f, 4.f);
        std::vector<float> ref(n); for (int64_t i=0;i<n;i++) ref[i]=tanh_gelu(hx[i]);
        float *dx,*dy; CUDA_CHECK(cudaMalloc(&dx,n*4)); CUDA_CHECK(cudaMalloc(&dy,n*4));
        CUDA_CHECK(cudaMemcpy(dx,hx.data(),n*4,cudaMemcpyHostToDevice));
        gelu_forward_sm89_kernel<float><<<(n+511)/512,512>>>(dx,dy,n);
        const char* err=sync_err();
        report("gelu_forward<float>", err?1e9:max_abs_err(ref,dy,n), TOL_EW, err);
        cudaFree(dx);cudaFree(dy);
    }

    // ---- gelu_backward (float), OwnTensor::cuda ---- GPT MLP hidden gradient.
    {
        int64_t n = GPT_ROWS * GPT_4C; auto hg=rand_vec(n,2,-1.f,1.f); auto hx=rand_vec(n,3,-4.f,4.f);
        auto dgelu=[&](float x){ float h=1e-3f; return (tanh_gelu(x+h)-tanh_gelu(x-h))/(2*h); };
        std::vector<float> ref(n); for(int64_t i=0;i<n;i++) ref[i]=hg[i]*dgelu(hx[i]);
        float *dg,*dxi,*dgi; CUDA_CHECK(cudaMalloc(&dg,n*4));CUDA_CHECK(cudaMalloc(&dxi,n*4));CUDA_CHECK(cudaMalloc(&dgi,n*4));
        CUDA_CHECK(cudaMemcpy(dg,hg.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dxi,hx.data(),n*4,cudaMemcpyHostToDevice));
        OwnTensor::cuda::gelu_backward_sm89_kernel<float><<<(n+511)/512,512>>>(dg,dxi,dgi,n);
        const char* err=sync_err();
        // analytic-vs-finite-diff slack: use a looser tol for this row.
        report("gelu_backward<float>", err?1e9:max_abs_err(ref,dgi,n), 5e-2, err);
        cudaFree(dg);cudaFree(dxi);cudaFree(dgi);
    }

    // ---- layer_norm_forward<float,float> ---- GPT: rows=B*T tokens, cols=C.
    {
        int rows=(int)GPT_ROWS, cols=GPT_C; int64_t n=(int64_t)rows*cols;
        auto hx=rand_vec(n,4); auto hg=rand_vec(cols,5); auto hb=rand_vec(cols,6);
        std::vector<float> refy(n); float eps=1e-5f;
        for(int r=0;r<rows;r++){
            double m=0; for(int c=0;c<cols;c++) m+=hx[r*cols+c]; m/=cols;
            double var=0; for(int c=0;c<cols;c++){double d=hx[r*cols+c]-m; var+=d*d;} var/=cols;
            double rs=1.0/std::sqrt(var+eps);
            for(int c=0;c<cols;c++) refy[r*cols+c]=(float)(((hx[r*cols+c]-m)*rs)*hg[c]+hb[c]);
        }
        float *dx,*dg,*db,*dy,*dm,*drs;
        CUDA_CHECK(cudaMalloc(&dx,n*4));CUDA_CHECK(cudaMalloc(&dg,cols*4));CUDA_CHECK(cudaMalloc(&db,cols*4));
        CUDA_CHECK(cudaMalloc(&dy,n*4));CUDA_CHECK(cudaMalloc(&dm,rows*4));CUDA_CHECK(cudaMalloc(&drs,rows*4));
        CUDA_CHECK(cudaMemcpy(dx,hx.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dg,hg.data(),cols*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db,hb.data(),cols*4,cudaMemcpyHostToDevice));
        layer_norm_forward_sm89_kernel<float,float><<<rows,512>>>(dx,dg,db,dy,dm,drs,cols,eps);
        const char* err=sync_err();
        report("layer_norm_forward<float,float>", err?1e9:max_abs_err(refy,dy,n), TOL_EW, err);
        cudaFree(dx);cudaFree(dg);cudaFree(db);cudaFree(dy);cudaFree(dm);cudaFree(drs);
    }

    // ---- ln_backward_input<float>, OwnTensor::cuda ---- GPT: rows=B*T, cols=C.
    {
        int rows=(int)GPT_ROWS, cols=GPT_C; int64_t n=(int64_t)rows*cols;
        auto hdy=rand_vec(n,7); auto hx=rand_vec(n,8); auto hg=rand_vec(cols,9);
        std::vector<float> hm(rows),hrs(rows); float eps=1e-5f;
        for(int r=0;r<rows;r++){
            double m=0; for(int c=0;c<cols;c++) m+=hx[r*cols+c]; m/=cols;
            double var=0; for(int c=0;c<cols;c++){double d=hx[r*cols+c]-m;var+=d*d;} var/=cols;
            hm[r]=(float)m; hrs[r]=(float)(1.0/std::sqrt(var+eps));
        }
        std::vector<float> ref(n);
        for(int r=0;r<rows;r++){
            double s1=0,s2=0;
            for(int c=0;c<cols;c++){double nx=(hx[r*cols+c]-hm[r])*hrs[r];double a=hdy[r*cols+c]*hg[c];s1+=a;s2+=a*nx;}
            double ic=1.0/cols;
            for(int c=0;c<cols;c++){double nx=(hx[r*cols+c]-hm[r])*hrs[r];
                ref[r*cols+c]=(float)(hrs[r]*(hdy[r*cols+c]*hg[c]-(s1+nx*s2)*ic));}
        }
        float *ddy,*dx,*dm,*drs,*dg,*dgx;
        CUDA_CHECK(cudaMalloc(&ddy,n*4));CUDA_CHECK(cudaMalloc(&dx,n*4));CUDA_CHECK(cudaMalloc(&dm,rows*4));
        CUDA_CHECK(cudaMalloc(&drs,rows*4));CUDA_CHECK(cudaMalloc(&dg,cols*4));CUDA_CHECK(cudaMalloc(&dgx,n*4));
        CUDA_CHECK(cudaMemcpy(ddy,hdy.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx,hx.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dm,hm.data(),rows*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(drs,hrs.data(),rows*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dg,hg.data(),cols*4,cudaMemcpyHostToDevice));
        OwnTensor::cuda::ln_backward_input_sm89_kernel<float><<<rows,512>>>(ddy,dx,dm,drs,dg,dgx,cols);
        const char* err=sync_err();
        report("ln_backward_input<float>", err?1e9:max_abs_err(ref,dgx,n), TOL_EW, err);
        cudaFree(ddy);cudaFree(dx);cudaFree(dm);cudaFree(drs);cudaFree(dg);cudaFree(dgx);
    }

    // ---- ln_backward_gamma_beta<float> (grad_gamma) ---- GPT: rows=B*T, cols=C.
    {
        int rows=(int)GPT_ROWS, cols=GPT_C; int64_t n=(int64_t)rows*cols;
        auto hdy=rand_vec(n,10); auto hx=rand_vec(n,11); float eps=1e-5f;
        std::vector<float> hm(rows),hrs(rows);
        for(int r=0;r<rows;r++){double m=0;for(int c=0;c<cols;c++)m+=hx[r*cols+c];m/=cols;
            double var=0;for(int c=0;c<cols;c++){double d=hx[r*cols+c]-m;var+=d*d;}var/=cols;
            hm[r]=(float)m;hrs[r]=(float)(1.0/std::sqrt(var+eps));}
        std::vector<float> refdg(cols,0.f), refdb(cols,0.f);
        for(int c=0;c<cols;c++){double dg=0,dbb=0;
            for(int r=0;r<rows;r++){double nx=(hx[r*cols+c]-hm[r])*hrs[r];dg+=hdy[r*cols+c]*nx;dbb+=hdy[r*cols+c];}
            refdg[c]=(float)dg;refdb[c]=(float)dbb;}
        float *ddy,*dx,*dm,*drs,*dgg,*dgb;
        CUDA_CHECK(cudaMalloc(&ddy,n*4));CUDA_CHECK(cudaMalloc(&dx,n*4));CUDA_CHECK(cudaMalloc(&dm,rows*4));
        CUDA_CHECK(cudaMalloc(&drs,rows*4));CUDA_CHECK(cudaMalloc(&dgg,cols*4));CUDA_CHECK(cudaMalloc(&dgb,cols*4));
        CUDA_CHECK(cudaMemcpy(ddy,hdy.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx,hx.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dm,hm.data(),rows*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(drs,hrs.data(),rows*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dgg,0,cols*4));CUDA_CHECK(cudaMemset(dgb,0,cols*4));
        dim3 blk(32,8), grd((cols+31)/32,(rows+7)/8);
        ln_backward_gamma_beta_sm89_kernel<float><<<grd,blk>>>(ddy,dx,dm,drs,dgg,dgb,rows,cols);
        const char* err=sync_err();
        report("ln_backward_gamma_beta<float>(dgamma)", err?1e9:max_abs_err(refdg,dgg,cols), TOL_EW, err);
        cudaFree(ddy);cudaFree(dx);cudaFree(dm);cudaFree(drs);cudaFree(dgg);cudaFree(dgb);
    }

    // ---- sparse_ce_forward + sparseCENormalize (chained), float / int32 ----
    // GPT logits: rows=B*T tokens, vocab=50304. (CPU reference loops rows*vocab ~ 0.8B
    // exp() calls, so this row takes a few seconds; it is the slowest case in the harness.)
    {
        int rows=(int)GPT_ROWS; int vocab=GPT_VOCAB; int64_t n=(int64_t)rows*vocab;
        auto hl=rand_vec(n,12,-3.f,3.f);
        std::vector<int32_t> ht(rows); for(int r=0;r<rows;r++) ht[r]=(r*37)%vocab;
        // CPU reference: per-row loss and softmax max/sum
        std::vector<float> refloss(rows), refmax(rows), refsum(rows);
        for(int r=0;r<rows;r++){
            float mx=-1e38f; for(int c=0;c<vocab;c++) mx=std::max(mx,hl[r*vocab+c]);
            double sum=0; for(int c=0;c<vocab;c++) sum+=std::exp(hl[r*vocab+c]-mx);
            refmax[r]=mx; refsum[r]=(float)sum;
            refloss[r]=(float)(std::log(sum)-(hl[r*vocab+ht[r]]-mx));
        }
        float *dl,*dloss,*dmax,*dsum; int32_t* dt;
        CUDA_CHECK(cudaMalloc(&dl,n*4));CUDA_CHECK(cudaMalloc(&dt,rows*4));
        CUDA_CHECK(cudaMalloc(&dloss,rows*4));CUDA_CHECK(cudaMalloc(&dmax,rows*4));CUDA_CHECK(cudaMalloc(&dsum,rows*4));
        CUDA_CHECK(cudaMemcpy(dl,hl.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dt,ht.data(),rows*4,cudaMemcpyHostToDevice));
        int bdim=256; size_t smem=(size_t)(bdim*4 + bdim + bdim)*sizeof(float);
        sparse_ce_forward_kernel_vec_save_stats<float,int32_t><<<rows,bdim,smem>>>(
            dl,dt,dloss,dmax,dsum,rows,vocab);
        const char* err1=sync_err();
        report("sparse_ce_forward<float,int32>(loss)", err1?1e9:max_abs_err(refloss,dloss,rows), TOL_EW, err1);

        // backward grad = (softmax - onehot) * scale
        float scale=1.0f; std::vector<float> hgo(1,1.0f);
        std::vector<float> refgrad(n);
        for(int r=0;r<rows;r++){double inv=1.0/refsum[r];
            for(int c=0;c<vocab;c++){float sm=(float)(std::exp(hl[r*vocab+c]-refmax[r])*inv);
                refgrad[r*vocab+c]=(sm-(c==ht[r]?1.0f:0.0f))*scale;}}
        float *dgrad,*dgo;
        CUDA_CHECK(cudaMalloc(&dgrad,n*4));CUDA_CHECK(cudaMalloc(&dgo,4));
        CUDA_CHECK(cudaMemcpy(dgo,hgo.data(),4,cudaMemcpyHostToDevice));
        sparseCENormalize_from_stats<float,int32_t><<<rows,bdim>>>(
            dl,dt,dgrad,dmax,dsum,rows,vocab,dgo,scale);
        const char* err2=sync_err();
        report("sparseCENormalize<float,int32>(grad)", err2?1e9:max_abs_err(refgrad,dgrad,n), TOL_EW, err2);
        cudaFree(dl);cudaFree(dt);cudaFree(dloss);cudaFree(dmax);cudaFree(dsum);cudaFree(dgrad);cudaFree(dgo);
    }

    // ---- multi_tensor_adam (GPT param list) ----
    // GPT-124M has ~148 parameter tensors, but the linked kernel's AdamLaunchMetadata ABI
    // caps one launch at MTA_MAXT=48 tensors / MTA_MAXB=320 chunks (chunk=32768 -> ~10.5M
    // elements). A full 124M-param model needs many such launches; here we test one launch
    // over a representative GPT-shaped slice: a repeating cycle of the model's real tensor
    // shapes (LN gain/bias = C, attn/MLP bias = C or 4C), filling up to the ABI caps.
    {
        // Real GPT-124M per-block tensor sizes (the small/medium ones that fit the chunk cap).
        const int64_t shapes[] = { GPT_C, GPT_C, GPT_4C, GPT_C, GPT_C, GPT_4C };
        const int n_shapes = (int)(sizeof(shapes)/sizeof(shapes[0]));
        std::vector<int64_t> sizes; int64_t total=0; int chunks=0;
        for (int t=0; t<MTA_MAXT; ++t) {
            int64_t sz = shapes[t % n_shapes];
            int c = (int)((sz + MTA_CHUNK - 1) / MTA_CHUNK);
            if (chunks + c > MTA_MAXB) break;           // respect the block-table cap
            sizes.push_back(sz); total += sz; chunks += c;
        }
        const int n_tensors = (int)sizes.size();

        float lr=1e-3f,b1=0.9f,b2=0.999f,eps=1e-8f,wd=0.01f,bc1=1.f-b1,bc2=1.f-b2;
        AdamLaunchMetadata meta{};
        std::vector<float*> dps, dgs, dms, dvs;           // all kept alive until after launch
        std::vector<std::vector<float>> refs(n_tensors);
        int blk_cursor = 0;
        for (int t=0; t<n_tensors; ++t) {
            int64_t n = sizes[t];
            auto hp=rand_vec(n,(uint64_t)(13+4*t)); auto hg=rand_vec(n,(uint64_t)(14+4*t));
            auto hm=rand_vec(n,(uint64_t)(15+4*t),0.f,0.1f); auto hv=rand_vec(n,(uint64_t)(16+4*t),0.f,0.1f);
            refs[t].resize(n);
            for(int64_t i=0;i<n;i++){float m=fmaf(b1,hm[i],(1-b1)*hg[i]);float v=fmaf(b2,hv[i],(1-b2)*hg[i]*hg[i]);
                float denom=std::sqrt(v/bc2)+eps;refs[t][i]=hp[i]-lr*((m/bc1)/denom+wd*hp[i]);}
            float *dp,*dg,*dm,*dv;
            CUDA_CHECK(cudaMalloc(&dp,n*4));CUDA_CHECK(cudaMalloc(&dg,n*4));CUDA_CHECK(cudaMalloc(&dm,n*4));CUDA_CHECK(cudaMalloc(&dv,n*4));
            CUDA_CHECK(cudaMemcpy(dp,hp.data(),n*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dg,hg.data(),n*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dm,hm.data(),n*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dv,hv.data(),n*4,cudaMemcpyHostToDevice));
            meta.params[t]=dp;meta.grads[t]=dg;meta.ms[t]=dm;meta.vs[t]=dv;meta.sizes[t]=n;
            dps.push_back(dp);dgs.push_back(dg);dms.push_back(dm);dvs.push_back(dv);
            int c=(int)((n+MTA_CHUNK-1)/MTA_CHUNK);
            for(int j=0;j<c;j++){meta.block_to_tensor[blk_cursor]=(unsigned char)t;meta.block_to_chunk[blk_cursor]=j;blk_cursor++;}
        }
        multi_tensor_adam_sm89_kernel<<<blk_cursor,256>>>(meta,lr,b1,b2,eps,wd,bc1,bc2);
        const char* err=sync_err();
        double emax=0;
        if(!err) for(int t=0;t<n_tensors;++t) emax=std::max(emax, max_abs_err(refs[t], dps[t], sizes[t]));
        char label[96];
        snprintf(label,sizeof(label),"multi_tensor_adam(%d tensors, %d chunks)", n_tensors, blk_cursor);
        report(label, err?1e9:emax, TOL_EW, err);
        for(int t=0;t<n_tensors;++t){cudaFree(dps[t]);cudaFree(dgs[t]);cudaFree(dms[t]);cudaFree(dvs[t]);}
    }

    // ---- mem_efficient_bwd_precompute_D<64> ---- GPT attention: B, heads, T, head_dim=64.
    // D[row] = sum_k dO[row,k] * O[row,k], one row per (batch, head, token).
    {
        int B=GPT_B,nh=GPT_HEADS,T=GPT_T,Hd=GPT_HD; int64_t rows=(int64_t)B*nh*T; int64_t n=rows*Hd;
        auto hdo=rand_vec(n,17); auto ho=rand_vec(n,18);
        std::vector<float> refD(rows);
        for(int64_t r=0;r<rows;r++){double s=0;for(int k=0;k<Hd;k++)s+=hdo[r*Hd+k]*ho[r*Hd+k];refD[r]=(float)s;}
        float *ddo,*do_o,*dD;
        CUDA_CHECK(cudaMalloc(&ddo,n*4));CUDA_CHECK(cudaMalloc(&do_o,n*4));CUDA_CHECK(cudaMalloc(&dD,rows*4));
        CUDA_CHECK(cudaMemcpy(ddo,hdo.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(do_o,ho.data(),n*4,cudaMemcpyHostToDevice));
        MemEfficientBwdParams p{};
        p.dO=ddo; p.O=do_o; p.D=dD; p.B=B; p.nh=nh; p.T=T; p.scale=1.f; p.is_causal=false;
        p.do_strideB=(int64_t)nh*T*Hd; p.do_strideH=(int64_t)T*Hd; p.do_strideM=Hd;
        p.o_strideB=p.do_strideB; p.o_strideH=p.do_strideH; p.o_strideM=Hd;
        p.d_strideB=(int64_t)nh*T; p.d_strideH=T;
        // grid: blockIdx.y = B*nh; blockIdx.x tiles rows in groups of (BLOCK_M_D*2)=16; 256 threads.
        dim3 grd((T + 15)/16, B*nh), blk(256);
        mem_efficient_bwd_precompute_D_sm89<64><<<grd,blk>>>(p);
        const char* err=sync_err();
        report("precompute_D<64>", err?1e9:max_abs_err(refD,dD,rows), TOL_TF, err);
        cudaFree(ddo);cudaFree(do_o);cudaFree(dD);
    }

    // ---- fused_attn_forward + mem_efficient_bwd_unified (FlashAttention fwd/bwd) ----
    // Driven at attention-shaped tensors with head_dim=64 (the only compiled instantiation).
    // B/T are reduced from full GPT (16x1024) because the CPU reference is O(B*H*T^2*Hd);
    // T=256 still exercises multiple Q tiles (BQ=64), many KV tiles (BK=64 fwd / 16 bwd),
    // and the causal mask. Tolerances are TF32 relative (scaled to peak magnitude) since
    // both kernels accumulate through TF32 tensor cores; tighten them once calibrated on
    // the target sm_89 GPU.
    {
        const int AB = 2, AH = 6, AT = 256, AHD = 64;
        const float ascale = 1.0f / std::sqrt((float)AHD);
        const int64_t arows = (int64_t)AB * AH * AT;     // (b,h,token) rows
        const int64_t an     = arows * AHD;              // Q/K/V/O/dO element count
        // contiguous [B,H,T,Hd] / [B,H,T] strides
        const int64_t sB = (int64_t)AH * AT * AHD, sH = (int64_t)AT * AHD, sM = AHD;
        const int64_t lB = (int64_t)AH * AT,       lH = AT;

        auto hQ = rand_vec(an, 101), hK = rand_vec(an, 102), hV = rand_vec(an, 103);

        float *dQ_, *dK_, *dV_, *dO_, *dLSE_;
        CUDA_CHECK(cudaMalloc(&dQ_, an*4)); CUDA_CHECK(cudaMalloc(&dK_, an*4));
        CUDA_CHECK(cudaMalloc(&dV_, an*4)); CUDA_CHECK(cudaMalloc(&dO_, an*4));
        CUDA_CHECK(cudaMalloc(&dLSE_, arows*4));
        CUDA_CHECK(cudaMemcpy(dQ_, hQ.data(), an*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK_, hK.data(), an*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV_, hV.data(), an*4, cudaMemcpyHostToDevice));

        // forward shared-memory footprint (mirrors the kernel's smem layout, BQ=BK=64).
        const int HD_PAD = AHD + 4, BK_STRIDE = 64 + 8;
        size_t fwd_smem = sizeof(float) * ((size_t)64*HD_PAD + 2*64*HD_PAD + 64*BK_STRIDE + 3*64);
        CUDA_CHECK(cudaFuncSetAttribute(
            fused_attn_forward_kernel_tc_sm89<64,64,64,1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fwd_smem));

        for (int causal = 0; causal <= 1; ++causal) {
            std::vector<float> refO(an), refLSE(arows);
            attn_ref_forward(hQ, hK, hV, AB, AH, AT, AHD, ascale, causal, refO, refLSE);

            MemEfficientFwdParams fp{};
            fp.Q=dQ_; fp.K=dK_; fp.V=dV_; fp.O=dO_; fp.LSE=dLSE_;
            fp.B=AB; fp.nh=AH; fp.T=AT; fp.scale=ascale; fp.is_causal=(bool)causal;
            fp.dropout_p=0.f; fp.dropout_mask=nullptr;
            fp.q_strideB=sB; fp.q_strideH=sH; fp.q_strideM=sM;
            fp.k_strideB=sB; fp.k_strideH=sH; fp.k_strideM=sM;
            fp.v_strideB=sB; fp.v_strideH=sH; fp.v_strideM=sM;
            fp.o_strideB=sB; fp.o_strideH=sH; fp.o_strideM=sM;
            fp.lse_strideB=lB; fp.lse_strideH=lH;
            dim3 fgrd((AT + 63)/64, AB*AH), fblk(256);
            fused_attn_forward_kernel_tc_sm89<64,64,64,1><<<fgrd, fblk, fwd_smem>>>(fp);
            const char* ferr = sync_err();
            const char* tag = causal ? "(causal)" : "(noncausal)";
            char lbl[64];
            snprintf(lbl,sizeof(lbl),"fused_attn_forward<64> O %s", tag);
            report(lbl, ferr?1e9:max_abs_err(refO,dO_,an), std::max(2e-3,0.04*vmax_abs(refO)), ferr);
            snprintf(lbl,sizeof(lbl),"fused_attn_forward<64> LSE %s", tag);
            report(lbl, ferr?1e9:max_abs_err(refLSE,dLSE_,arows), std::max(2e-3,0.02*vmax_abs(refLSE)), ferr);
        }

        // ---- backward: feed correct O/LSE/D, check dQ/dK/dV ----
        // backward shared-memory footprint (mirrors the kernel's smem layout).
        const int BKN_PAD = 16 + 4, BM_PAD = 32 + 4;
        size_t bwd_smem = sizeof(float) * ((size_t)2*32*HD_PAD + 2*16*HD_PAD
                          + 2*32*BKN_PAD + 2*16*BM_PAD + 32*HD_PAD + 2*32);
        auto hdO = rand_vec(an, 104);
        float *bdO, *bO, *bD, *bdQ, *bdK, *bdV;
        CUDA_CHECK(cudaMalloc(&bdO, an*4)); CUDA_CHECK(cudaMalloc(&bO, an*4));
        CUDA_CHECK(cudaMalloc(&bD, arows*4));
        CUDA_CHECK(cudaMalloc(&bdQ, an*4)); CUDA_CHECK(cudaMalloc(&bdK, an*4)); CUDA_CHECK(cudaMalloc(&bdV, an*4));
        CUDA_CHECK(cudaMemcpy(bdO, hdO.data(), an*4, cudaMemcpyHostToDevice));

        for (int causal = 0; causal <= 1; ++causal) {
            std::vector<float> refO(an), refLSE(arows);
            attn_ref_forward(hQ, hK, hV, AB, AH, AT, AHD, ascale, causal, refO, refLSE);
            std::vector<float> refD(arows), refdQ(an), refdK(an), refdV(an);
            attn_ref_backward(hQ, hK, hV, refO, hdO, refLSE, AB, AH, AT, AHD, ascale, causal,
                              refD, refdQ, refdK, refdV);

            CUDA_CHECK(cudaMemcpy(bO, refO.data(), an*4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(bD, refD.data(), arows*4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dLSE_, refLSE.data(), arows*4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(bdQ, 0, an*4));   // dQ is accumulated via atomicAdd
            CUDA_CHECK(cudaMemset(bdK, 0, an*4));
            CUDA_CHECK(cudaMemset(bdV, 0, an*4));

            OwnTensor::MemEfficientBwdParams bp{};
            bp.Q=dQ_; bp.K=dK_; bp.V=dV_; bp.O=bO; bp.dO=bdO; bp.LSE=dLSE_;
            bp.D=bD; bp.dQ=bdQ; bp.dK=bdK; bp.dV=bdV;
            bp.B=AB; bp.nh=AH; bp.T=AT; bp.scale=ascale; bp.is_causal=(bool)causal;
            bp.q_strideB=sB; bp.q_strideH=sH; bp.q_strideM=sM;
            bp.k_strideB=sB; bp.k_strideH=sH; bp.k_strideM=sM;
            bp.v_strideB=sB; bp.v_strideH=sH; bp.v_strideM=sM;
            bp.o_strideB=sB; bp.o_strideH=sH; bp.o_strideM=sM;
            bp.do_strideB=sB; bp.do_strideH=sH; bp.do_strideM=sM;
            bp.dq_strideB=sB; bp.dq_strideH=sH; bp.dq_strideM=sM;
            bp.dk_strideB=sB; bp.dk_strideH=sH; bp.dk_strideM=sM;
            bp.dv_strideB=sB; bp.dv_strideH=sH; bp.dv_strideM=sM;
            bp.lse_strideB=lB; bp.lse_strideH=lH;
            bp.d_strideB=lB; bp.d_strideH=lH;
            dim3 bgrd((AT + 15)/16, AB*AH), bblk(256);
            if (causal)
                OwnTensor::mem_efficient_bwd_unified_kernel_exp12<64,true ><<<bgrd,bblk,bwd_smem>>>(bp);
            else
                OwnTensor::mem_efficient_bwd_unified_kernel_exp12<64,false><<<bgrd,bblk,bwd_smem>>>(bp);
            const char* berr = sync_err();
            const char* tag = causal ? "(causal)" : "(noncausal)";
            char lbl[64];
            snprintf(lbl,sizeof(lbl),"bwd_unified<64> dQ %s", tag);
            report(lbl, berr?1e9:max_abs_err(refdQ,bdQ,an), std::max(2e-3,0.08*vmax_abs(refdQ)), berr);
            snprintf(lbl,sizeof(lbl),"bwd_unified<64> dK %s", tag);
            report(lbl, berr?1e9:max_abs_err(refdK,bdK,an), std::max(2e-3,0.08*vmax_abs(refdK)), berr);
            snprintf(lbl,sizeof(lbl),"bwd_unified<64> dV %s", tag);
            report(lbl, berr?1e9:max_abs_err(refdV,bdV,an), std::max(2e-3,0.08*vmax_abs(refdV)), berr);
        }

        cudaFree(dQ_); cudaFree(dK_); cudaFree(dV_); cudaFree(dO_); cudaFree(dLSE_);
        cudaFree(bdO); cudaFree(bO); cudaFree(bD); cudaFree(bdQ); cudaFree(bdK); cudaFree(bdV);
    }

    printf("\nNote: mem_efficient_bwd_precompute_D is covered above; the backward harness\n");
    printf("feeds a CPU-computed D so the unified bwd kernel is checked in isolation.\n");
    printf("unified_reduce_standalone: covered by its own self-test binary.\n\n");

    printf("%s  (%d failures)\n", g_fail==0 ? "ALL PASSED" : "SOME FAILED", g_fail);
    return g_fail==0 ? 0 : 1;
}
