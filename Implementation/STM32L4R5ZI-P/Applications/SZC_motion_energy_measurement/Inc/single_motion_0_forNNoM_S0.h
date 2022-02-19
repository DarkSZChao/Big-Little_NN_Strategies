#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0_S0 {-53, -74, -52, -2, 57, -5, 57, 29, 62, 85, -58, -6, -1, -24, -24, 22, -42, 28, 11, 55, -19, 19, 26, 18, 40, 53, 24, -18, 124, 26, -36, -15, 8, 1, -21, -27}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS_S0 {7}

#define TENSOR_CONV1D_BIAS_0_S0 {45, -42, 63, -30}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS_S0 {7}

#define CONV1D_BIAS_LSHIFT_S0 {0}

#define CONV1D_OUTPUT_RSHIFT_S0 {8}

#define TENSOR_CONV1D_1_KERNEL_0_S0 {-3, 92, -28, 36, -40, 39, 34, 84, -57, -7, -37, 10, 52, -65, -52, 22, 52, 25, 11, 12, 42, 27, 19, -50, 7, 48, -15, 52, 36, -28, -48, -22, 69, -8, 9, -17, 34, -71, -18, -15, 37, 12, -36, 5, -89, -53, -33, 64}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S0 {7}

#define TENSOR_CONV1D_1_BIAS_0_S0 {-65, 105, 40, -66}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S0 {7}

#define CONV1D_1_BIAS_LSHIFT_S0 {0}

#define CONV1D_1_OUTPUT_RSHIFT_S0 {7}

#define TENSOR_CONV1D_2_KERNEL_0_S0 {32, -27, 34, 14, -29, 57, -60, -98, -27, -7, 12, -23, 42, 0, -15, -27, -24, -36, -27, -13, 12, -90, -24, 107}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S0 {7}

#define TENSOR_CONV1D_2_BIAS_0_S0 {-18, -42}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S0 {7}

#define CONV1D_2_BIAS_LSHIFT_S0 {0}

#define CONV1D_2_OUTPUT_RSHIFT_S0 {6}

#define TENSOR_DENSE_KERNEL_0_S0 {33, 44, -24, 33, -33, 49, -60, 35, -10, 8, -31, 51, 15, 56, 0, 85, -50, 54, -14, 36, 1, 72, -54, 73, -26, 87, -19, 82, -9, 47, -23, 8, -33, -43, -1, -44, 8, -9, -35, -43, 2, -51, -1, -46, 53, -45, 59, -39, 3, -56, 46, -53, 36, -14, 14, -8, 29, -9, 38, -23, 35, -61, -11, -25}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS_S0 {7}

#define TENSOR_DENSE_BIAS_0_S0 {126, -126}

#define TENSOR_DENSE_BIAS_0_DEC_BITS_S0 {7}

#define DENSE_BIAS_LSHIFT_S0 {0}

#define DENSE_OUTPUT_RSHIFT_S0 {8}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC_S0 0
#define INPUT_1_OUTPUT_OFFSET_S0 0
#define CONV1D_OUTPUT_DEC_S0 -1
#define CONV1D_OUTPUT_OFFSET_S0 0
#define RE_LU_OUTPUT_DEC_S0 -1
#define RE_LU_OUTPUT_OFFSET_S0 0
#define MAX_POOLING1D_OUTPUT_DEC_S0 -1
#define MAX_POOLING1D_OUTPUT_OFFSET_S0 0
#define CONV1D_1_OUTPUT_DEC_S0 -1
#define CONV1D_1_OUTPUT_OFFSET_S0 0
#define RE_LU_1_OUTPUT_DEC_S0 -1
#define RE_LU_1_OUTPUT_OFFSET_S0 0
#define MAX_POOLING1D_1_OUTPUT_DEC_S0 -1
#define MAX_POOLING1D_1_OUTPUT_OFFSET_S0 0
#define CONV1D_2_OUTPUT_DEC_S0 0
#define CONV1D_2_OUTPUT_OFFSET_S0 0
#define RE_LU_2_OUTPUT_DEC_S0 0
#define RE_LU_2_OUTPUT_OFFSET_S0 0
#define MAX_POOLING1D_2_OUTPUT_DEC_S0 0
#define MAX_POOLING1D_2_OUTPUT_OFFSET_S0 0
#define FLATTEN_OUTPUT_DEC_S0 0
#define FLATTEN_OUTPUT_OFFSET_S0 0
#define DENSE_OUTPUT_DEC_S0 -1
#define DENSE_OUTPUT_OFFSET_S0 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data_S0[384] = {0};

const nnom_shape_data_t tensor_input_1_0_dim_S0[] = {128, 3};
const nnom_qformat_param_t tensor_input_1_0_dec_S0[] = {0};
const nnom_qformat_param_t tensor_input_1_0_offset_S0[] = {0};
const nnom_tensor_t tensor_input_1_0_S0 = {
    .p_data = (void*)nnom_input_data_S0,
    .dim = (nnom_shape_data_t*)tensor_input_1_0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config_S0 = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1_0_S0
};
const int8_t tensor_conv1d_kernel_0_data_S0[] = TENSOR_CONV1D_KERNEL_0_S0;

const nnom_shape_data_t tensor_conv1d_kernel_0_dim_S0[] = {3, 3, 4};
const nnom_qformat_param_t tensor_conv1d_kernel_0_dec_S0[] = TENSOR_CONV1D_KERNEL_0_DEC_BITS_S0;
const nnom_qformat_param_t tensor_conv1d_kernel_0_offset_S0[] = {0};
const nnom_tensor_t tensor_conv1d_kernel_0_S0 = {
    .p_data = (void*)tensor_conv1d_kernel_0_data_S0,
    .dim = (nnom_shape_data_t*)tensor_conv1d_kernel_0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_bias_0_data_S0[] = TENSOR_CONV1D_BIAS_0_S0;

const nnom_shape_data_t tensor_conv1d_bias_0_dim_S0[] = {4};
const nnom_qformat_param_t tensor_conv1d_bias_0_dec_S0[] = TENSOR_CONV1D_BIAS_0_DEC_BITS_S0;
const nnom_qformat_param_t tensor_conv1d_bias_0_offset_S0[] = {0};
const nnom_tensor_t tensor_conv1d_bias_0_S0 = {
    .p_data = (void*)tensor_conv1d_bias_0_data_S0,
    .dim = (nnom_shape_data_t*)tensor_conv1d_bias_0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_bias_0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_bias_0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_output_shift_S0[] = CONV1D_OUTPUT_RSHIFT_S0;
const nnom_qformat_param_t conv1d_bias_shift_S0[] = CONV1D_BIAS_LSHIFT_S0;
const nnom_conv2d_config_t conv1d_config_S0 = {
    .super = {.name = "conv1d"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_kernel_0_S0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_bias_0_S0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_output_shift_S0, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_bias_shift_S0, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_config_S0 = {
    .super = {.name = "max_pooling1d"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_1_kernel_0_data_S0[] = TENSOR_CONV1D_1_KERNEL_0_S0;

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim_S0[] = {3, 4, 4};
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_dec_S0[] = TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S0;
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_offset_S0[] = {0};
const nnom_tensor_t tensor_conv1d_1_kernel_0_S0 = {
    .p_data = (void*)tensor_conv1d_1_kernel_0_data_S0,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_kernel_0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_1_bias_0_data_S0[] = TENSOR_CONV1D_1_BIAS_0_S0;

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim_S0[] = {4};
const nnom_qformat_param_t tensor_conv1d_1_bias_0_dec_S0[] = TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S0;
const nnom_qformat_param_t tensor_conv1d_1_bias_0_offset_S0[] = {0};
const nnom_tensor_t tensor_conv1d_1_bias_0_S0 = {
    .p_data = (void*)tensor_conv1d_1_bias_0_data_S0,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_bias_0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_1_output_shift_S0[] = CONV1D_1_OUTPUT_RSHIFT_S0;
const nnom_qformat_param_t conv1d_1_bias_shift_S0[] = CONV1D_1_BIAS_LSHIFT_S0;
const nnom_conv2d_config_t conv1d_1_config_S0 = {
    .super = {.name = "conv1d_1"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_1_kernel_0_S0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_1_bias_0_S0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_1_output_shift_S0, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_1_bias_shift_S0, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_1_config_S0 = {
    .super = {.name = "max_pooling1d_1"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_2_kernel_0_data_S0[] = TENSOR_CONV1D_2_KERNEL_0_S0;

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim_S0[] = {3, 4, 2};
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_dec_S0[] = TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S0;
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_offset_S0[] = {0};
const nnom_tensor_t tensor_conv1d_2_kernel_0_S0 = {
    .p_data = (void*)tensor_conv1d_2_kernel_0_data_S0,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_kernel_0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_2_bias_0_data_S0[] = TENSOR_CONV1D_2_BIAS_0_S0;

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim_S0[] = {2};
const nnom_qformat_param_t tensor_conv1d_2_bias_0_dec_S0[] = TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S0;
const nnom_qformat_param_t tensor_conv1d_2_bias_0_offset_S0[] = {0};
const nnom_tensor_t tensor_conv1d_2_bias_0_S0 = {
    .p_data = (void*)tensor_conv1d_2_bias_0_data_S0,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_bias_0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_2_output_shift_S0[] = CONV1D_2_OUTPUT_RSHIFT_S0;
const nnom_qformat_param_t conv1d_2_bias_shift_S0[] = CONV1D_2_BIAS_LSHIFT_S0;
const nnom_conv2d_config_t conv1d_2_config_S0 = {
    .super = {.name = "conv1d_2"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_2_kernel_0_S0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_2_bias_0_S0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_2_output_shift_S0, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_2_bias_shift_S0, 
    .filter_size = 2,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_2_config_S0 = {
    .super = {.name = "max_pooling1d_2"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};

const nnom_flatten_config_t flatten_config_S0 = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data_S0[] = TENSOR_DENSE_KERNEL_0_S0;

const nnom_shape_data_t tensor_dense_kernel_0_dim_S0[] = {32, 2};
const nnom_qformat_param_t tensor_dense_kernel_0_dec_S0[] = TENSOR_DENSE_KERNEL_0_DEC_BITS_S0;
const nnom_qformat_param_t tensor_dense_kernel_0_offset_S0[] = {0};
const nnom_tensor_t tensor_dense_kernel_0_S0 = {
    .p_data = (void*)tensor_dense_kernel_0_data_S0,
    .dim = (nnom_shape_data_t*)tensor_dense_kernel_0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_kernel_0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_kernel_0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_bias_0_data_S0[] = TENSOR_DENSE_BIAS_0_S0;

const nnom_shape_data_t tensor_dense_bias_0_dim_S0[] = {2};
const nnom_qformat_param_t tensor_dense_bias_0_dec_S0[] = TENSOR_DENSE_BIAS_0_DEC_BITS_S0;
const nnom_qformat_param_t tensor_dense_bias_0_offset_S0[] = {0};
const nnom_tensor_t tensor_dense_bias_0_S0 = {
    .p_data = (void*)tensor_dense_bias_0_data_S0,
    .dim = (nnom_shape_data_t*)tensor_dense_bias_0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_bias_0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_bias_0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_output_shift_S0[] = DENSE_OUTPUT_RSHIFT_S0;
const nnom_qformat_param_t dense_bias_shift_S0[] = DENSE_BIAS_LSHIFT_S0;
const nnom_dense_config_t dense_config_S0 = {
    .super = {.name = "dense"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_kernel_0_S0,
    .bias = (nnom_tensor_t*)&tensor_dense_bias_0_S0,
    .output_shift = (nnom_qformat_param_t *)&dense_output_shift_S0,
    .bias_shift = (nnom_qformat_param_t *)&dense_bias_shift_S0
};
static int8_t nnom_output_data_S0[2] = {0};

const nnom_shape_data_t tensor_output0_dim_S0[] = {2};
const nnom_qformat_param_t tensor_output0_dec_S0[] = {DENSE_OUTPUT_DEC_S0};
const nnom_qformat_param_t tensor_output0_offset_S0[] = {0};
const nnom_tensor_t tensor_output0_S0 = {
    .p_data = (void*)nnom_output_data_S0,
    .dim = (nnom_shape_data_t*)tensor_output0_dim_S0,
    .q_dec = (nnom_qformat_param_t*)tensor_output0_dec_S0,
    .q_offset = (nnom_qformat_param_t*)tensor_output0_offset_S0,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output0_config_S0 = {
    .super = {.name = "output0"},
    .tensor = (nnom_tensor_t*)&tensor_output0_S0
};
/* model version */
#define NNOM_MODEL_VERSION_S0 (10000*0 + 100*4 + 3)

/* nnom model */
static nnom_model_t* nnom_model_create_S0(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[13];

	check_model_version(NNOM_MODEL_VERSION_S0);
	new_model(&model);

	layer[0] = input_s(&input_1_config_S0);
	layer[1] = model.hook(conv2d_s(&conv1d_config_S0), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(maxpool_s(&max_pooling1d_config_S0), layer[2]);
	layer[4] = model.hook(conv2d_s(&conv1d_1_config_S0), layer[3]);
	layer[5] = model.active(act_relu(), layer[4]);
	layer[6] = model.hook(maxpool_s(&max_pooling1d_1_config_S0), layer[5]);
	layer[7] = model.hook(conv2d_s(&conv1d_2_config_S0), layer[6]);
	layer[8] = model.active(act_relu(), layer[7]);
	layer[9] = model.hook(maxpool_s(&max_pooling1d_2_config_S0), layer[8]);
	layer[10] = model.hook(flatten_s(&flatten_config_S0), layer[9]);
	layer[11] = model.hook(dense_s(&dense_config_S0), layer[10]);
	layer[12] = model.hook(output_s(&output0_config_S0), layer[11]);
	model_compile(&model, layer[0], layer[12]);
	return &model;
}
