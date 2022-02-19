#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0_S5 {63, -62, 54, -20, -33, 41, 22, -25, 19, -17, -45, -14, -40, 19, 30, -23, -11, -60, 78, 46, 48, -2, -20, 36, 28, 57, -7, -71, 59, 23, 18, 8, 1, -41, 25, 7}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS_S5 {7}

#define TENSOR_CONV1D_BIAS_0_S5 {1, -12, 29, -10}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS_S5 {7}

#define CONV1D_BIAS_LSHIFT_S5 {0}

#define CONV1D_OUTPUT_RSHIFT_S5 {8}

#define TENSOR_CONV1D_1_KERNEL_0_S5 {47, 17, 39, 21, -59, -48, -44, 43, 30, -31, -10, 17, -8, 67, -27, 13, -53, 67, -27, -34, 40, 22, -54, 46, -13, 37, 83, -3, 47, -28, 30, -36, -11, 40, 28, -71, 67, 57, 40, -57, -46, -55, 70, 25, -7, 46, -49, -37}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S5 {7}

#define TENSOR_CONV1D_1_BIAS_0_S5 {-20, -47, 87, 76}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S5 {7}

#define CONV1D_1_BIAS_LSHIFT_S5 {0}

#define CONV1D_1_OUTPUT_RSHIFT_S5 {7}

#define TENSOR_CONV1D_2_KERNEL_0_S5 {-50, -61, 7, -10, 52, 63, 62, 69, -11, -35, 5, -2, -29, -6, 20, -34, -52, -35, -74, -58, -27, 17, -21, 7}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S5 {7}

#define TENSOR_CONV1D_2_BIAS_0_S5 {67, -2}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S5 {7}

#define CONV1D_2_BIAS_LSHIFT_S5 {0}

#define CONV1D_2_OUTPUT_RSHIFT_S5 {7}

#define TENSOR_DENSE_KERNEL_0_S5 {-48, 22, 4, -26, 47, 16, -16, -1, 21, -32, 79, -1, 17, -52, 0, 0, 3, -36, 25, 17, 24, -1, 56, 1, 22, 8, 6, 4, 39, -10, 53, -31, 8, 32, -57, -5, -73, -2, -1, -38, -22, -24, -26, 4, 20, -9, 3, -25, 1, -35, -51, 28, -14, 36, -30, -19, -53, 32, -62, -26, -58, -30, -31, -5}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS_S5 {7}

#define TENSOR_DENSE_BIAS_0_S5 {-63, 63}

#define TENSOR_DENSE_BIAS_0_DEC_BITS_S5 {7}

#define DENSE_BIAS_LSHIFT_S5 {0}

#define DENSE_OUTPUT_RSHIFT_S5 {8}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC_S5 0
#define INPUT_1_OUTPUT_OFFSET_S5 0
#define CONV1D_OUTPUT_DEC_S5 -1
#define CONV1D_OUTPUT_OFFSET_S5 0
#define RE_LU_OUTPUT_DEC_S5 -1
#define RE_LU_OUTPUT_OFFSET_S5 0
#define MAX_POOLING1D_OUTPUT_DEC_S5 -1
#define MAX_POOLING1D_OUTPUT_OFFSET_S5 0
#define CONV1D_1_OUTPUT_DEC_S5 -1
#define CONV1D_1_OUTPUT_OFFSET_S5 0
#define RE_LU_1_OUTPUT_DEC_S5 -1
#define RE_LU_1_OUTPUT_OFFSET_S5 0
#define MAX_POOLING1D_1_OUTPUT_DEC_S5 -1
#define MAX_POOLING1D_1_OUTPUT_OFFSET_S5 0
#define CONV1D_2_OUTPUT_DEC_S5 -1
#define CONV1D_2_OUTPUT_OFFSET_S5 0
#define RE_LU_2_OUTPUT_DEC_S5 -1
#define RE_LU_2_OUTPUT_OFFSET_S5 0
#define MAX_POOLING1D_2_OUTPUT_DEC_S5 -1
#define MAX_POOLING1D_2_OUTPUT_OFFSET_S5 0
#define FLATTEN_OUTPUT_DEC_S5 -1
#define FLATTEN_OUTPUT_OFFSET_S5 0
#define DENSE_OUTPUT_DEC_S5 -2
#define DENSE_OUTPUT_OFFSET_S5 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data_S5[384] = {0};

const nnom_shape_data_t tensor_input_1_0_dim_S5[] = {128, 3};
const nnom_qformat_param_t tensor_input_1_0_dec_S5[] = {0};
const nnom_qformat_param_t tensor_input_1_0_offset_S5[] = {0};
const nnom_tensor_t tensor_input_1_0_S5 = {
    .p_data = (void*)nnom_input_data_S5,
    .dim = (nnom_shape_data_t*)tensor_input_1_0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config_S5 = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1_0_S5
};
const int8_t tensor_conv1d_kernel_0_data_S5[] = TENSOR_CONV1D_KERNEL_0_S5;

const nnom_shape_data_t tensor_conv1d_kernel_0_dim_S5[] = {3, 3, 4};
const nnom_qformat_param_t tensor_conv1d_kernel_0_dec_S5[] = TENSOR_CONV1D_KERNEL_0_DEC_BITS_S5;
const nnom_qformat_param_t tensor_conv1d_kernel_0_offset_S5[] = {0};
const nnom_tensor_t tensor_conv1d_kernel_0_S5 = {
    .p_data = (void*)tensor_conv1d_kernel_0_data_S5,
    .dim = (nnom_shape_data_t*)tensor_conv1d_kernel_0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_bias_0_data_S5[] = TENSOR_CONV1D_BIAS_0_S5;

const nnom_shape_data_t tensor_conv1d_bias_0_dim_S5[] = {4};
const nnom_qformat_param_t tensor_conv1d_bias_0_dec_S5[] = TENSOR_CONV1D_BIAS_0_DEC_BITS_S5;
const nnom_qformat_param_t tensor_conv1d_bias_0_offset_S5[] = {0};
const nnom_tensor_t tensor_conv1d_bias_0_S5 = {
    .p_data = (void*)tensor_conv1d_bias_0_data_S5,
    .dim = (nnom_shape_data_t*)tensor_conv1d_bias_0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_bias_0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_bias_0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_output_shift_S5[] = CONV1D_OUTPUT_RSHIFT_S5;
const nnom_qformat_param_t conv1d_bias_shift_S5[] = CONV1D_BIAS_LSHIFT_S5;
const nnom_conv2d_config_t conv1d_config_S5 = {
    .super = {.name = "conv1d"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_kernel_0_S5,
    .bias = (nnom_tensor_t*)&tensor_conv1d_bias_0_S5,
    .output_shift = (nnom_qformat_param_t *)&conv1d_output_shift_S5, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_bias_shift_S5, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_config_S5 = {
    .super = {.name = "max_pooling1d"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_1_kernel_0_data_S5[] = TENSOR_CONV1D_1_KERNEL_0_S5;

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim_S5[] = {3, 4, 4};
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_dec_S5[] = TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S5;
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_offset_S5[] = {0};
const nnom_tensor_t tensor_conv1d_1_kernel_0_S5 = {
    .p_data = (void*)tensor_conv1d_1_kernel_0_data_S5,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_kernel_0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_1_bias_0_data_S5[] = TENSOR_CONV1D_1_BIAS_0_S5;

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim_S5[] = {4};
const nnom_qformat_param_t tensor_conv1d_1_bias_0_dec_S5[] = TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S5;
const nnom_qformat_param_t tensor_conv1d_1_bias_0_offset_S5[] = {0};
const nnom_tensor_t tensor_conv1d_1_bias_0_S5 = {
    .p_data = (void*)tensor_conv1d_1_bias_0_data_S5,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_bias_0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_1_output_shift_S5[] = CONV1D_1_OUTPUT_RSHIFT_S5;
const nnom_qformat_param_t conv1d_1_bias_shift_S5[] = CONV1D_1_BIAS_LSHIFT_S5;
const nnom_conv2d_config_t conv1d_1_config_S5 = {
    .super = {.name = "conv1d_1"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_1_kernel_0_S5,
    .bias = (nnom_tensor_t*)&tensor_conv1d_1_bias_0_S5,
    .output_shift = (nnom_qformat_param_t *)&conv1d_1_output_shift_S5, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_1_bias_shift_S5, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_1_config_S5 = {
    .super = {.name = "max_pooling1d_1"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_2_kernel_0_data_S5[] = TENSOR_CONV1D_2_KERNEL_0_S5;

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim_S5[] = {3, 4, 2};
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_dec_S5[] = TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S5;
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_offset_S5[] = {0};
const nnom_tensor_t tensor_conv1d_2_kernel_0_S5 = {
    .p_data = (void*)tensor_conv1d_2_kernel_0_data_S5,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_kernel_0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_2_bias_0_data_S5[] = TENSOR_CONV1D_2_BIAS_0_S5;

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim_S5[] = {2};
const nnom_qformat_param_t tensor_conv1d_2_bias_0_dec_S5[] = TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S5;
const nnom_qformat_param_t tensor_conv1d_2_bias_0_offset_S5[] = {0};
const nnom_tensor_t tensor_conv1d_2_bias_0_S5 = {
    .p_data = (void*)tensor_conv1d_2_bias_0_data_S5,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_bias_0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_2_output_shift_S5[] = CONV1D_2_OUTPUT_RSHIFT_S5;
const nnom_qformat_param_t conv1d_2_bias_shift_S5[] = CONV1D_2_BIAS_LSHIFT_S5;
const nnom_conv2d_config_t conv1d_2_config_S5 = {
    .super = {.name = "conv1d_2"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_2_kernel_0_S5,
    .bias = (nnom_tensor_t*)&tensor_conv1d_2_bias_0_S5,
    .output_shift = (nnom_qformat_param_t *)&conv1d_2_output_shift_S5, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_2_bias_shift_S5, 
    .filter_size = 2,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_2_config_S5 = {
    .super = {.name = "max_pooling1d_2"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};

const nnom_flatten_config_t flatten_config_S5 = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data_S5[] = TENSOR_DENSE_KERNEL_0_S5;

const nnom_shape_data_t tensor_dense_kernel_0_dim_S5[] = {32, 2};
const nnom_qformat_param_t tensor_dense_kernel_0_dec_S5[] = TENSOR_DENSE_KERNEL_0_DEC_BITS_S5;
const nnom_qformat_param_t tensor_dense_kernel_0_offset_S5[] = {0};
const nnom_tensor_t tensor_dense_kernel_0_S5 = {
    .p_data = (void*)tensor_dense_kernel_0_data_S5,
    .dim = (nnom_shape_data_t*)tensor_dense_kernel_0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_kernel_0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_kernel_0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_bias_0_data_S5[] = TENSOR_DENSE_BIAS_0_S5;

const nnom_shape_data_t tensor_dense_bias_0_dim_S5[] = {2};
const nnom_qformat_param_t tensor_dense_bias_0_dec_S5[] = TENSOR_DENSE_BIAS_0_DEC_BITS_S5;
const nnom_qformat_param_t tensor_dense_bias_0_offset_S5[] = {0};
const nnom_tensor_t tensor_dense_bias_0_S5 = {
    .p_data = (void*)tensor_dense_bias_0_data_S5,
    .dim = (nnom_shape_data_t*)tensor_dense_bias_0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_bias_0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_bias_0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_output_shift_S5[] = DENSE_OUTPUT_RSHIFT_S5;
const nnom_qformat_param_t dense_bias_shift_S5[] = DENSE_BIAS_LSHIFT_S5;
const nnom_dense_config_t dense_config_S5 = {
    .super = {.name = "dense"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_kernel_0_S5,
    .bias = (nnom_tensor_t*)&tensor_dense_bias_0_S5,
    .output_shift = (nnom_qformat_param_t *)&dense_output_shift_S5,
    .bias_shift = (nnom_qformat_param_t *)&dense_bias_shift_S5
};
static int8_t nnom_output_data_S5[2] = {0};

const nnom_shape_data_t tensor_output0_dim_S5[] = {2};
const nnom_qformat_param_t tensor_output0_dec_S5[] = {DENSE_OUTPUT_DEC_S5};
const nnom_qformat_param_t tensor_output0_offset_S5[] = {0};
const nnom_tensor_t tensor_output0_S5 = {
    .p_data = (void*)nnom_output_data_S5,
    .dim = (nnom_shape_data_t*)tensor_output0_dim_S5,
    .q_dec = (nnom_qformat_param_t*)tensor_output0_dec_S5,
    .q_offset = (nnom_qformat_param_t*)tensor_output0_offset_S5,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output0_config_S5 = {
    .super = {.name = "output0"},
    .tensor = (nnom_tensor_t*)&tensor_output0_S5
};
/* model version */
#define NNOM_MODEL_VERSION_S5 (10000*0 + 100*4 + 3)

/* nnom model */
static nnom_model_t* nnom_model_create_S5(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[13];

	check_model_version(NNOM_MODEL_VERSION_S5);
	new_model(&model);

	layer[0] = input_s(&input_1_config_S5);
	layer[1] = model.hook(conv2d_s(&conv1d_config_S5), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(maxpool_s(&max_pooling1d_config_S5), layer[2]);
	layer[4] = model.hook(conv2d_s(&conv1d_1_config_S5), layer[3]);
	layer[5] = model.active(act_relu(), layer[4]);
	layer[6] = model.hook(maxpool_s(&max_pooling1d_1_config_S5), layer[5]);
	layer[7] = model.hook(conv2d_s(&conv1d_2_config_S5), layer[6]);
	layer[8] = model.active(act_relu(), layer[7]);
	layer[9] = model.hook(maxpool_s(&max_pooling1d_2_config_S5), layer[8]);
	layer[10] = model.hook(flatten_s(&flatten_config_S5), layer[9]);
	layer[11] = model.hook(dense_s(&dense_config_S5), layer[10]);
	layer[12] = model.hook(output_s(&output0_config_S5), layer[11]);
	model_compile(&model, layer[0], layer[12]);
	return &model;
}
