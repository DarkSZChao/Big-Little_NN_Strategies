#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0_S4 {27, -49, -98, -39, 54, 23, 6, -47, -6, -47, 21, -4, -40, -18, -2, -11, 46, -94, -91, 65, -2, -8, 73, -3, 9, -8, 41, -102, 37, -8, -10, 13, -41, 37, -42, 59}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS_S4 {7}

#define TENSOR_CONV1D_BIAS_0_S4 {-25, 46, -16, 22}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS_S4 {7}

#define CONV1D_BIAS_LSHIFT_S4 {0}

#define CONV1D_OUTPUT_RSHIFT_S4 {8}

#define TENSOR_CONV1D_1_KERNEL_0_S4 {-80, 24, 26, 59, 49, 43, 4, 39, 22, -80, -75, 30, -51, -31, -34, 13, 4, -44, -70, 25, 45, 69, 29, 48, 17, -18, -69, -2, 23, 25, -49, 45, -68, 51, -38, 53, -46, -18, -11, 19, -14, 1, -4, 60, 37, 56, -5, 61}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S4 {7}

#define TENSOR_CONV1D_1_BIAS_0_S4 {-27, -26, 26, -9}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S4 {7}

#define CONV1D_1_BIAS_LSHIFT_S4 {0}

#define CONV1D_1_OUTPUT_RSHIFT_S4 {7}

#define TENSOR_CONV1D_2_KERNEL_0_S4 {-30, -47, -16, 25, 59, 23, -27, 38, -80, -40, -91, 47, 55, 15, -57, 60, -15, 65, -20, 21, -23, -11, -12, 16}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S4 {7}

#define TENSOR_CONV1D_2_BIAS_0_S4 {-15, -44}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S4 {7}

#define CONV1D_2_BIAS_LSHIFT_S4 {0}

#define CONV1D_2_OUTPUT_RSHIFT_S4 {7}

#define TENSOR_DENSE_KERNEL_0_S4 {-49, -70, 59, 4, -21, 5, 24, -34, 13, 47, -24, 101, -36, 42, 7, 25, -19, 38, 53, 30, 8, 69, -15, -4, 24, -25, -4, 74, 27, 24, 58, 70, 55, 16, 32, -31, -30, -18, 20, -24, 38, -38, -19, -14, -1, 44, 13, 0, -21, -12, -56, -35, -16, -50, -32, -4, -43, -50, -29, -42, 11, 5, -87, 12}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS_S4 {7}

#define TENSOR_DENSE_BIAS_0_S4 {-34, 34}

#define TENSOR_DENSE_BIAS_0_DEC_BITS_S4 {7}

#define DENSE_BIAS_LSHIFT_S4 {0}

#define DENSE_OUTPUT_RSHIFT_S4 {8}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC_S4 0
#define INPUT_1_OUTPUT_OFFSET_S4 0
#define CONV1D_OUTPUT_DEC_S4 -1
#define CONV1D_OUTPUT_OFFSET_S4 0
#define RE_LU_OUTPUT_DEC_S4 -1
#define RE_LU_OUTPUT_OFFSET_S4 0
#define MAX_POOLING1D_OUTPUT_DEC_S4 -1
#define MAX_POOLING1D_OUTPUT_OFFSET_S4 0
#define CONV1D_1_OUTPUT_DEC_S4 -1
#define CONV1D_1_OUTPUT_OFFSET_S4 0
#define RE_LU_1_OUTPUT_DEC_S4 -1
#define RE_LU_1_OUTPUT_OFFSET_S4 0
#define MAX_POOLING1D_1_OUTPUT_DEC_S4 -1
#define MAX_POOLING1D_1_OUTPUT_OFFSET_S4 0
#define CONV1D_2_OUTPUT_DEC_S4 -1
#define CONV1D_2_OUTPUT_OFFSET_S4 0
#define RE_LU_2_OUTPUT_DEC_S4 -1
#define RE_LU_2_OUTPUT_OFFSET_S4 0
#define MAX_POOLING1D_2_OUTPUT_DEC_S4 -1
#define MAX_POOLING1D_2_OUTPUT_OFFSET_S4 0
#define FLATTEN_OUTPUT_DEC_S4 -1
#define FLATTEN_OUTPUT_OFFSET_S4 0
#define DENSE_OUTPUT_DEC_S4 -2
#define DENSE_OUTPUT_OFFSET_S4 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data_S4[384] = {0};

const nnom_shape_data_t tensor_input_1_0_dim_S4[] = {128, 3};
const nnom_qformat_param_t tensor_input_1_0_dec_S4[] = {0};
const nnom_qformat_param_t tensor_input_1_0_offset_S4[] = {0};
const nnom_tensor_t tensor_input_1_0_S4 = {
    .p_data = (void*)nnom_input_data_S4,
    .dim = (nnom_shape_data_t*)tensor_input_1_0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config_S4 = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1_0_S4
};
const int8_t tensor_conv1d_kernel_0_data_S4[] = TENSOR_CONV1D_KERNEL_0_S4;

const nnom_shape_data_t tensor_conv1d_kernel_0_dim_S4[] = {3, 3, 4};
const nnom_qformat_param_t tensor_conv1d_kernel_0_dec_S4[] = TENSOR_CONV1D_KERNEL_0_DEC_BITS_S4;
const nnom_qformat_param_t tensor_conv1d_kernel_0_offset_S4[] = {0};
const nnom_tensor_t tensor_conv1d_kernel_0_S4 = {
    .p_data = (void*)tensor_conv1d_kernel_0_data_S4,
    .dim = (nnom_shape_data_t*)tensor_conv1d_kernel_0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_bias_0_data_S4[] = TENSOR_CONV1D_BIAS_0_S4;

const nnom_shape_data_t tensor_conv1d_bias_0_dim_S4[] = {4};
const nnom_qformat_param_t tensor_conv1d_bias_0_dec_S4[] = TENSOR_CONV1D_BIAS_0_DEC_BITS_S4;
const nnom_qformat_param_t tensor_conv1d_bias_0_offset_S4[] = {0};
const nnom_tensor_t tensor_conv1d_bias_0_S4 = {
    .p_data = (void*)tensor_conv1d_bias_0_data_S4,
    .dim = (nnom_shape_data_t*)tensor_conv1d_bias_0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_bias_0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_bias_0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_output_shift_S4[] = CONV1D_OUTPUT_RSHIFT_S4;
const nnom_qformat_param_t conv1d_bias_shift_S4[] = CONV1D_BIAS_LSHIFT_S4;
const nnom_conv2d_config_t conv1d_config_S4 = {
    .super = {.name = "conv1d"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_kernel_0_S4,
    .bias = (nnom_tensor_t*)&tensor_conv1d_bias_0_S4,
    .output_shift = (nnom_qformat_param_t *)&conv1d_output_shift_S4, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_bias_shift_S4, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_config_S4 = {
    .super = {.name = "max_pooling1d"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_1_kernel_0_data_S4[] = TENSOR_CONV1D_1_KERNEL_0_S4;

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim_S4[] = {3, 4, 4};
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_dec_S4[] = TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S4;
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_offset_S4[] = {0};
const nnom_tensor_t tensor_conv1d_1_kernel_0_S4 = {
    .p_data = (void*)tensor_conv1d_1_kernel_0_data_S4,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_kernel_0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_1_bias_0_data_S4[] = TENSOR_CONV1D_1_BIAS_0_S4;

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim_S4[] = {4};
const nnom_qformat_param_t tensor_conv1d_1_bias_0_dec_S4[] = TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S4;
const nnom_qformat_param_t tensor_conv1d_1_bias_0_offset_S4[] = {0};
const nnom_tensor_t tensor_conv1d_1_bias_0_S4 = {
    .p_data = (void*)tensor_conv1d_1_bias_0_data_S4,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_bias_0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_1_output_shift_S4[] = CONV1D_1_OUTPUT_RSHIFT_S4;
const nnom_qformat_param_t conv1d_1_bias_shift_S4[] = CONV1D_1_BIAS_LSHIFT_S4;
const nnom_conv2d_config_t conv1d_1_config_S4 = {
    .super = {.name = "conv1d_1"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_1_kernel_0_S4,
    .bias = (nnom_tensor_t*)&tensor_conv1d_1_bias_0_S4,
    .output_shift = (nnom_qformat_param_t *)&conv1d_1_output_shift_S4, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_1_bias_shift_S4, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_1_config_S4 = {
    .super = {.name = "max_pooling1d_1"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_2_kernel_0_data_S4[] = TENSOR_CONV1D_2_KERNEL_0_S4;

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim_S4[] = {3, 4, 2};
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_dec_S4[] = TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S4;
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_offset_S4[] = {0};
const nnom_tensor_t tensor_conv1d_2_kernel_0_S4 = {
    .p_data = (void*)tensor_conv1d_2_kernel_0_data_S4,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_kernel_0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_2_bias_0_data_S4[] = TENSOR_CONV1D_2_BIAS_0_S4;

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim_S4[] = {2};
const nnom_qformat_param_t tensor_conv1d_2_bias_0_dec_S4[] = TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S4;
const nnom_qformat_param_t tensor_conv1d_2_bias_0_offset_S4[] = {0};
const nnom_tensor_t tensor_conv1d_2_bias_0_S4 = {
    .p_data = (void*)tensor_conv1d_2_bias_0_data_S4,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_bias_0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_2_output_shift_S4[] = CONV1D_2_OUTPUT_RSHIFT_S4;
const nnom_qformat_param_t conv1d_2_bias_shift_S4[] = CONV1D_2_BIAS_LSHIFT_S4;
const nnom_conv2d_config_t conv1d_2_config_S4 = {
    .super = {.name = "conv1d_2"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_2_kernel_0_S4,
    .bias = (nnom_tensor_t*)&tensor_conv1d_2_bias_0_S4,
    .output_shift = (nnom_qformat_param_t *)&conv1d_2_output_shift_S4, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_2_bias_shift_S4, 
    .filter_size = 2,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_2_config_S4 = {
    .super = {.name = "max_pooling1d_2"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};

const nnom_flatten_config_t flatten_config_S4 = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data_S4[] = TENSOR_DENSE_KERNEL_0_S4;

const nnom_shape_data_t tensor_dense_kernel_0_dim_S4[] = {32, 2};
const nnom_qformat_param_t tensor_dense_kernel_0_dec_S4[] = TENSOR_DENSE_KERNEL_0_DEC_BITS_S4;
const nnom_qformat_param_t tensor_dense_kernel_0_offset_S4[] = {0};
const nnom_tensor_t tensor_dense_kernel_0_S4 = {
    .p_data = (void*)tensor_dense_kernel_0_data_S4,
    .dim = (nnom_shape_data_t*)tensor_dense_kernel_0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_kernel_0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_kernel_0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_bias_0_data_S4[] = TENSOR_DENSE_BIAS_0_S4;

const nnom_shape_data_t tensor_dense_bias_0_dim_S4[] = {2};
const nnom_qformat_param_t tensor_dense_bias_0_dec_S4[] = TENSOR_DENSE_BIAS_0_DEC_BITS_S4;
const nnom_qformat_param_t tensor_dense_bias_0_offset_S4[] = {0};
const nnom_tensor_t tensor_dense_bias_0_S4 = {
    .p_data = (void*)tensor_dense_bias_0_data_S4,
    .dim = (nnom_shape_data_t*)tensor_dense_bias_0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_bias_0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_bias_0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_output_shift_S4[] = DENSE_OUTPUT_RSHIFT_S4;
const nnom_qformat_param_t dense_bias_shift_S4[] = DENSE_BIAS_LSHIFT_S4;
const nnom_dense_config_t dense_config_S4 = {
    .super = {.name = "dense"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_kernel_0_S4,
    .bias = (nnom_tensor_t*)&tensor_dense_bias_0_S4,
    .output_shift = (nnom_qformat_param_t *)&dense_output_shift_S4,
    .bias_shift = (nnom_qformat_param_t *)&dense_bias_shift_S4
};
static int8_t nnom_output_data_S4[2] = {0};

const nnom_shape_data_t tensor_output0_dim_S4[] = {2};
const nnom_qformat_param_t tensor_output0_dec_S4[] = {DENSE_OUTPUT_DEC_S4};
const nnom_qformat_param_t tensor_output0_offset_S4[] = {0};
const nnom_tensor_t tensor_output0_S4 = {
    .p_data = (void*)nnom_output_data_S4,
    .dim = (nnom_shape_data_t*)tensor_output0_dim_S4,
    .q_dec = (nnom_qformat_param_t*)tensor_output0_dec_S4,
    .q_offset = (nnom_qformat_param_t*)tensor_output0_offset_S4,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output0_config_S4 = {
    .super = {.name = "output0"},
    .tensor = (nnom_tensor_t*)&tensor_output0_S4
};
/* model version */
#define NNOM_MODEL_VERSION_S4 (10000*0 + 100*4 + 3)

/* nnom model */
static nnom_model_t* nnom_model_create_S4(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[13];

	check_model_version(NNOM_MODEL_VERSION_S4);
	new_model(&model);

	layer[0] = input_s(&input_1_config_S4);
	layer[1] = model.hook(conv2d_s(&conv1d_config_S4), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(maxpool_s(&max_pooling1d_config_S4), layer[2]);
	layer[4] = model.hook(conv2d_s(&conv1d_1_config_S4), layer[3]);
	layer[5] = model.active(act_relu(), layer[4]);
	layer[6] = model.hook(maxpool_s(&max_pooling1d_1_config_S4), layer[5]);
	layer[7] = model.hook(conv2d_s(&conv1d_2_config_S4), layer[6]);
	layer[8] = model.active(act_relu(), layer[7]);
	layer[9] = model.hook(maxpool_s(&max_pooling1d_2_config_S4), layer[8]);
	layer[10] = model.hook(flatten_s(&flatten_config_S4), layer[9]);
	layer[11] = model.hook(dense_s(&dense_config_S4), layer[10]);
	layer[12] = model.hook(output_s(&output0_config_S4), layer[11]);
	model_compile(&model, layer[0], layer[12]);
	return &model;
}
