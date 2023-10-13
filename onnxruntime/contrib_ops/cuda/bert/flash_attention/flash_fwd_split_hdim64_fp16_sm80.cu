// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#if USE_FLASH_ATTENTION

#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace onnxruntime {
namespace flash {

<<<<<<< HEAD
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream);
=======
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64>(Flash_fwd_params& params, cudaStream_t stream);
>>>>>>> aciddelgado/gqa_memeff_v2

}  // namespace flash
}  // namespace onnxruntime
#endif
