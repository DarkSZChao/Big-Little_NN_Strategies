#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0_S3 {16, 59, 51, -37, 21, -25, 4, 87, -46, -46, 7, -6, 68, 28, -69, -21, -31, 60, 35, -24, -30, -22, 45, -48, -121, 32, -16, -52, -3, 34, 14, -23, -12, -25, -23, -40}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS_S3 {7}

#define TENSOR_CONV1D_BIAS_0_S3 {-44, -40, 25, -15}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS_S3 {7}

#define CONV1D_BIAS_LSHIFT_S3 {0}

#define CONV1D_OUTPUT_RSHIFT_S3 {8}

#define TENSOR_CONV1D_1_KERNEL_0_S3 {-46, 45, -41, 44, -70, -67, 24, -6, -24, 43, 1, 34, 1, -1, 1, -35, -12, -2, 49, -39, 54, 31, 7, 50, -22, -5, 52, 69, -41, 32, -3, -20, 31, 19, 5, -6, -10, 9, -19, -11, -5, -16, 75, 62, -52, 43, -36, 25}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S3 {7}

#define TENSOR_CONV1D_1_BIAS_0_S3 {-14, -14, 29, -3}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S3 {7}

#define CONV1D_1_BIAS_LSHIFT_S3 {0}

#define CONV1D_1_OUTPUT_RSHIFT_S3 {7}

#define TENSOR_CONV1D_2_KERNEL_0_S3 {-6, 24, 66, 0, 24, -28, -62, 86, -7, -6, -8, -23, -25, -48, 38, -32, 35, -58, -53, 40, 36, -67, -47, -52}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S3 {7}

#define TENSOR_CONV1D_2_BIAS_0_S3 {63, -11}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S3 {7}

#define CONV1D_2_BIAS_LSHIFT_S3 {0}

#define CONV1D_2_OUTPUT_RSHIFT_S3 {7}

#define TENSOR_DENSE_KERNEL_0_S3 {35, 1, -11, -24, 64, 31, 59, 1, 0, -38, -2, -5, 38, -29, 51, 7, 69, 32, -7, 16, 8, -27, 56, -17, 38, 61, 79, -5, 25, -13, 91, -13, -109, -38, -2, 38, -19, -2, 6, 34, -14, -19, -45, 11, -36, -2, 5, 29, -39, -34, -20, -43, -76, 20, -15, 9, -4, 38, -22, 1, -4, -45, -45, 8}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS_S3 {7}

#define TENSOR_DENSE_BIAS_0_S3 {-110, 110}

#define TENSOR_DENSE_BIAS_0_DEC_BITS_S3 {7}

#define DENSE_BIAS_LSHIFT_S3 {0}

#define DENSE_OUTPUT_RSHIFT_S3 {6}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC_S3 0
#define INPUT_1_OUTPUT_OFFSET_S3 0
#define CONV1D_OUTPUT_DEC_S3 -1
#define CONV1D_OUTPUT_OFFSET_S3 0
#define RE_LU_OUTPUT_DEC_S3 -1
#define RE_LU_OUTPUT_OFFSET_S3 0
#define MAX_POOLING1D_OUTPUT_DEC_S3 -1
#define MAX_POOLING1D_OUTPUT_OFFSET_S3 0
#define CONV1D_1_OUTPUT_DEC_S3 -1
#define CONV1D_1_OUTPUT_OFFSET_S3 0
#define RE_LU_1_OUTPUT_DEC_S3 -1
#define RE_LU_1_OUTPUT_OFFSET_S3 0
#define MAX_POOLING1D_1_OUTPUT_DEC_S3 -1
#define MAX_POOLING1D_1_OUTPUT_OFFSET_S3 0
#define CONV1D_2_OUTPUT_DEC_S3 -1
#define CONV1D_2_OUTPUT_OFFSET_S3 0
#define RE_LU_2_OUTPUT_DEC_S3 -1
#define RE_LU_2_OUTPUT_OFFSET_S3 0
#define MAX_POOLING1D_2_OUTPUT_DEC_S3 -1
#define MAX_POOLING1D_2_OUTPUT_OFFSET_S3 0
#define FLATTEN_OUTPUT_DEC_S3 -1
#define FLATTEN_OUTPUT_OFFSET_S3 0
#define DENSE_OUTPUT_DEC_S3 0
#define DENSE_OUTPUT_OFFSET_S3 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data_S3[384] = {0};

const nnom_shape_data_t tensor_input_1_0_dim_S3[] = {128, 3};
const nnom_qformat_param_t tensor_input_1_0_dec_S3[] = {0};
const nnom_qformat_param_t tensor_input_1_0_offset_S3[] = {0};
const nnom_tensor_t tensor_input_1_0_S3 = {
    .p_data = (void*)nnom_input_data_S3,
    .dim = (nnom_shape_data_t*)tensor_input_1_0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config_S3 = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1_0_S3
};
const int8_t tensor_conv1d_kernel_0_data_S3[] = TENSOR_CONV1D_KERNEL_0_S3;

const nnom_shape_data_t tensor_conv1d_kernel_0_dim_S3[] = {3, 3, 4};
const nnom_qformat_param_t tensor_conv1d_kernel_0_dec_S3[] = TENSOR_CONV1D_KERNEL_0_DEC_BITS_S3;
const nnom_qformat_param_t tensor_conv1d_kernel_0_offset_S3[] = {0};
const nnom_tensor_t tensor_conv1d_kernel_0_S3 = {
    .p_data = (void*)tensor_conv1d_kernel_0_data_S3,
    .dim = (nnom_shape_data_t*)tensor_conv1d_kernel_0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_bias_0_data_S3[] = TENSOR_CONV1D_BIAS_0_S3;

const nnom_shape_data_t tensor_conv1d_bias_0_dim_S3[] = {4};
const nnom_qformat_param_t tensor_conv1d_bias_0_dec_S3[] = TENSOR_CONV1D_BIAS_0_DEC_BITS_S3;
const nnom_qformat_param_t tensor_conv1d_bias_0_offset_S3[] = {0};
const nnom_tensor_t tensor_conv1d_bias_0_S3 = {
    .p_data = (void*)tensor_conv1d_bias_0_data_S3,
    .dim = (nnom_shape_data_t*)tensor_conv1d_bias_0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_bias_0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_bias_0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_output_shift_S3[] = CONV1D_OUTPUT_RSHIFT_S3;
const nnom_qformat_param_t conv1d_bias_shift_S3[] = CONV1D_BIAS_LSHIFT_S3;
const nnom_conv2d_config_t conv1d_config_S3 = {
    .super = {.name = "conv1d"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_kernel_0_S3,
    .bias = (nnom_tensor_t*)&tensor_conv1d_bias_0_S3,
    .output_shift = (nnom_qformat_param_t *)&conv1d_output_shift_S3, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_bias_shift_S3, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_config_S3 = {
    .super = {.name = "max_pooling1d"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_1_kernel_0_data_S3[] = TENSOR_CONV1D_1_KERNEL_0_S3;

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim_S3[] = {3, 4, 4};
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_dec_S3[] = TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S3;
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_offset_S3[] = {0};
const nnom_tensor_t tensor_conv1d_1_kernel_0_S3 = {
    .p_data = (void*)tensor_conv1d_1_kernel_0_data_S3,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_kernel_0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_1_bias_0_data_S3[] = TENSOR_CONV1D_1_BIAS_0_S3;

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim_S3[] = {4};
const nnom_qformat_param_t tensor_conv1d_1_bias_0_dec_S3[] = TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S3;
const nnom_qformat_param_t tensor_conv1d_1_bias_0_offset_S3[] = {0};
const nnom_tensor_t tensor_conv1d_1_bias_0_S3 = {
    .p_data = (void*)tensor_conv1d_1_bias_0_data_S3,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_bias_0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_1_output_shift_S3[] = CONV1D_1_OUTPUT_RSHIFT_S3;
const nnom_qformat_param_t conv1d_1_bias_shift_S3[] = CONV1D_1_BIAS_LSHIFT_S3;
const nnom_conv2d_config_t conv1d_1_config_S3 = {
    .super = {.name = "conv1d_1"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_1_kernel_0_S3,
    .bias = (nnom_tensor_t*)&tensor_conv1d_1_bias_0_S3,
    .output_shift = (nnom_qformat_param_t *)&conv1d_1_output_shift_S3, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_1_bias_shift_S3, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_1_config_S3 = {
    .super = {.name = "max_pooling1d_1"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_2_kernel_0_data_S3[] = TENSOR_CONV1D_2_KERNEL_0_S3;

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim_S3[] = {3, 4, 2};
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_dec_S3[] = TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S3;
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_offset_S3[] = {0};
const nnom_tensor_t tensor_conv1d_2_kernel_0_S3 = {
    .p_data = (void*)tensor_conv1d_2_kernel_0_data_S3,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_kernel_0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_2_bias_0_data_S3[] = TENSOR_CONV1D_2_BIAS_0_S3;

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim_S3[] = {2};
const nnom_qformat_param_t tensor_conv1d_2_bias_0_dec_S3[] = TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S3;
const nnom_qformat_param_t tensor_conv1d_2_bias_0_offset_S3[] = {0};
const nnom_tensor_t tensor_conv1d_2_bias_0_S3 = {
    .p_data = (void*)tensor_conv1d_2_bias_0_data_S3,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_bias_0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_2_output_shift_S3[] = CONV1D_2_OUTPUT_RSHIFT_S3;
const nnom_qformat_param_t conv1d_2_bias_shift_S3[] = CONV1D_2_BIAS_LSHIFT_S3;
const nnom_conv2d_config_t conv1d_2_config_S3 = {
    .super = {.name = "conv1d_2"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_2_kernel_0_S3,
    .bias = (nnom_tensor_t*)&tensor_conv1d_2_bias_0_S3,
    .output_shift = (nnom_qformat_param_t *)&conv1d_2_output_shift_S3, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_2_bias_shift_S3, 
    .filter_size = 2,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_2_config_S3 = {
    .super = {.name = "max_pooling1d_2"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};

const nnom_flatten_config_t flatten_config_S3 = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data_S3[] = TENSOR_DENSE_KERNEL_0_S3;

const nnom_shape_data_t tensor_dense_kernel_0_dim_S3[] = {32, 2};
const nnom_qformat_param_t tensor_dense_kernel_0_dec_S3[] = TENSOR_DENSE_KERNEL_0_DEC_BITS_S3;
const nnom_qformat_param_t tensor_dense_kernel_0_offset_S3[] = {0};
const nnom_tensor_t tensor_dense_kernel_0_S3 = {
    .p_data = (void*)tensor_dense_kernel_0_data_S3,
    .dim = (nnom_shape_data_t*)tensor_dense_kernel_0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_kernel_0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_kernel_0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_bias_0_data_S3[] = TENSOR_DENSE_BIAS_0_S3;

const nnom_shape_data_t tensor_dense_bias_0_dim_S3[] = {2};
const nnom_qformat_param_t tensor_dense_bias_0_dec_S3[] = TENSOR_DENSE_BIAS_0_DEC_BITS_S3;
const nnom_qformat_param_t tensor_dense_bias_0_offset_S3[] = {0};
const nnom_tensor_t tensor_dense_bias_0_S3 = {
    .p_data = (void*)tensor_dense_bias_0_data_S3,
    .dim = (nnom_shape_data_t*)tensor_dense_bias_0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_bias_0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_bias_0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_output_shift_S3[] = DENSE_OUTPUT_RSHIFT_S3;
const nnom_qformat_param_t dense_bias_shift_S3[] = DENSE_BIAS_LSHIFT_S3;
const nnom_dense_config_t dense_config_S3 = {
    .super = {.name = "dense"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_kernel_0_S3,
    .bias = (nnom_tensor_t*)&tensor_dense_bias_0_S3,
    .output_shift = (nnom_qformat_param_t *)&dense_output_shift_S3,
    .bias_shift = (nnom_qformat_param_t *)&dense_bias_shift_S3
};
static int8_t nnom_output_data_S3[2] = {0};

const nnom_shape_data_t tensor_output0_dim_S3[] = {2};
const nnom_qformat_param_t tensor_output0_dec_S3[] = {DENSE_OUTPUT_DEC_S3};
const nnom_qformat_param_t tensor_output0_offset_S3[] = {0};
const nnom_tensor_t tensor_output0_S3 = {
    .p_data = (void*)nnom_output_data_S3,
    .dim = (nnom_shape_data_t*)tensor_output0_dim_S3,
    .q_dec = (nnom_qformat_param_t*)tensor_output0_dec_S3,
    .q_offset = (nnom_qformat_param_t*)tensor_output0_offset_S3,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output0_config_S3 = {
    .super = {.name = "output0"},
    .tensor = (nnom_tensor_t*)&tensor_output0_S3
};
/* model version */
#define NNOM_MODEL_VERSION_S3 (10000*0 + 100*4 + 3)

/* nnom model */
static nnom_model_t* nnom_model_create_S3(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[13];

	check_model_version(NNOM_MODEL_VERSION_S3);
	new_model(&model);

	layer[0] = input_s(&input_1_config_S3);
	layer[1] = model.hook(conv2d_s(&conv1d_config_S3), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(maxpool_s(&max_pooling1d_config_S3), layer[2]);
	layer[4] = model.hook(conv2d_s(&conv1d_1_config_S3), layer[3]);
	layer[5] = model.active(act_relu(), layer[4]);
	layer[6] = model.hook(maxpool_s(&max_pooling1d_1_config_S3), layer[5]);
	layer[7] = model.hook(conv2d_s(&conv1d_2_config_S3), layer[6]);
	layer[8] = model.active(act_relu(), layer[7]);
	layer[9] = model.hook(maxpool_s(&max_pooling1d_2_config_S3), layer[8]);
	layer[10] = model.hook(flatten_s(&flatten_config_S3), layer[9]);
	layer[11] = model.hook(dense_s(&dense_config_S3), layer[10]);
	layer[12] = model.hook(output_s(&output0_config_S3), layer[11]);
	model_compile(&model, layer[0], layer[12]);
	return &model;
}
