#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0 {63, -62, 54, -20, -33, 41, 22, -25, 19, -17, -45, -14, -40, 19, 30, -23, -11, -60, 78, 46, 48, -2, -20, 36, 28, 57, -7, -71, 59, 23, 18, 8, 1, -41, 25, 7}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS {7}

#define TENSOR_CONV1D_BIAS_0 {1, -12, 29, -10}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS {7}

#define CONV1D_BIAS_LSHIFT {0}

#define CONV1D_OUTPUT_RSHIFT {8}

#define TENSOR_CONV1D_1_KERNEL_0 {47, 17, 39, 21, -59, -48, -44, 43, 30, -31, -10, 17, -8, 67, -27, 13, -53, 67, -27, -34, 40, 22, -54, 46, -13, 37, 83, -3, 47, -28, 30, -36, -11, 40, 28, -71, 67, 57, 40, -57, -46, -55, 70, 25, -7, 46, -49, -37}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS {7}

#define TENSOR_CONV1D_1_BIAS_0 {-20, -47, 87, 76}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS {7}

#define CONV1D_1_BIAS_LSHIFT {0}

#define CONV1D_1_OUTPUT_RSHIFT {7}

#define TENSOR_CONV1D_2_KERNEL_0 {-50, -61, 7, -10, 52, 63, 62, 69, -11, -35, 5, -2, -29, -6, 20, -34, -52, -35, -74, -58, -27, 17, -21, 7}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS {7}

#define TENSOR_CONV1D_2_BIAS_0 {67, -2}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS {7}

#define CONV1D_2_BIAS_LSHIFT {0}

#define CONV1D_2_OUTPUT_RSHIFT {7}

#define TENSOR_DENSE_KERNEL_0 {-48, 22, 4, -26, 47, 16, -16, -1, 21, -32, 79, -1, 17, -52, 0, 0, 3, -36, 25, 17, 24, -1, 56, 1, 22, 8, 6, 4, 39, -10, 53, -31, 8, 32, -57, -5, -73, -2, -1, -38, -22, -24, -26, 4, 20, -9, 3, -25, 1, -35, -51, 28, -14, 36, -30, -19, -53, 32, -62, -26, -58, -30, -31, -5}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS {7}

#define TENSOR_DENSE_BIAS_0 {-63, 63}

#define TENSOR_DENSE_BIAS_0_DEC_BITS {7}

#define DENSE_BIAS_LSHIFT {0}

#define DENSE_OUTPUT_RSHIFT {8}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC 0
#define INPUT_1_OUTPUT_OFFSET 0
#define CONV1D_OUTPUT_DEC -1
#define CONV1D_OUTPUT_OFFSET 0
#define RE_LU_OUTPUT_DEC -1
#define RE_LU_OUTPUT_OFFSET 0
#define MAX_POOLING1D_OUTPUT_DEC -1
#define MAX_POOLING1D_OUTPUT_OFFSET 0
#define CONV1D_1_OUTPUT_DEC -1
#define CONV1D_1_OUTPUT_OFFSET 0
#define RE_LU_1_OUTPUT_DEC -1
#define RE_LU_1_OUTPUT_OFFSET 0
#define MAX_POOLING1D_1_OUTPUT_DEC -1
#define MAX_POOLING1D_1_OUTPUT_OFFSET 0
#define CONV1D_2_OUTPUT_DEC -1
#define CONV1D_2_OUTPUT_OFFSET 0
#define RE_LU_2_OUTPUT_DEC -1
#define RE_LU_2_OUTPUT_OFFSET 0
#define MAX_POOLING1D_2_OUTPUT_DEC -1
#define MAX_POOLING1D_2_OUTPUT_OFFSET 0
#define FLATTEN_OUTPUT_DEC -1
#define FLATTEN_OUTPUT_OFFSET 0
#define DENSE_OUTPUT_DEC -2
#define DENSE_OUTPUT_OFFSET 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data[384] = {0};

const nnom_shape_data_t tensor_input_1_0_dim[] = {128, 3};
const nnom_qformat_param_t tensor_input_1_0_dec[] = {0};
const nnom_qformat_param_t tensor_input_1_0_offset[] = {0};
const nnom_tensor_t tensor_input_1_0 = {
    .p_data = (void*)nnom_input_data,
    .dim = (nnom_shape_data_t*)tensor_input_1_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1_0
};
const int8_t tensor_conv1d_kernel_0_data[] = TENSOR_CONV1D_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_kernel_0_dim[] = {3, 3, 4};
const nnom_qformat_param_t tensor_conv1d_kernel_0_dec[] = TENSOR_CONV1D_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_kernel_0 = {
    .p_data = (void*)tensor_conv1d_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_bias_0_data[] = TENSOR_CONV1D_BIAS_0;

const nnom_shape_data_t tensor_conv1d_bias_0_dim[] = {4};
const nnom_qformat_param_t tensor_conv1d_bias_0_dec[] = TENSOR_CONV1D_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_bias_0 = {
    .p_data = (void*)tensor_conv1d_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_output_shift[] = CONV1D_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv1d_bias_shift[] = CONV1D_BIAS_LSHIFT;
const nnom_conv2d_config_t conv1d_config = {
    .super = {.name = "conv1d"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_bias_shift, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_config = {
    .super = {.name = "max_pooling1d"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_1_kernel_0_data[] = TENSOR_CONV1D_1_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim[] = {3, 4, 4};
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_dec[] = TENSOR_CONV1D_1_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_1_kernel_0 = {
    .p_data = (void*)tensor_conv1d_1_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_1_bias_0_data[] = TENSOR_CONV1D_1_BIAS_0;

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim[] = {4};
const nnom_qformat_param_t tensor_conv1d_1_bias_0_dec[] = TENSOR_CONV1D_1_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_1_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_1_bias_0 = {
    .p_data = (void*)tensor_conv1d_1_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_1_output_shift[] = CONV1D_1_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv1d_1_bias_shift[] = CONV1D_1_BIAS_LSHIFT;
const nnom_conv2d_config_t conv1d_1_config = {
    .super = {.name = "conv1d_1"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_1_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_1_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_1_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_1_bias_shift, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_1_config = {
    .super = {.name = "max_pooling1d_1"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_2_kernel_0_data[] = TENSOR_CONV1D_2_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim[] = {3, 4, 2};
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_dec[] = TENSOR_CONV1D_2_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_2_kernel_0 = {
    .p_data = (void*)tensor_conv1d_2_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_2_bias_0_data[] = TENSOR_CONV1D_2_BIAS_0;

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim[] = {2};
const nnom_qformat_param_t tensor_conv1d_2_bias_0_dec[] = TENSOR_CONV1D_2_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_2_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_2_bias_0 = {
    .p_data = (void*)tensor_conv1d_2_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_2_output_shift[] = CONV1D_2_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv1d_2_bias_shift[] = CONV1D_2_BIAS_LSHIFT;
const nnom_conv2d_config_t conv1d_2_config = {
    .super = {.name = "conv1d_2"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_2_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_2_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_2_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_2_bias_shift, 
    .filter_size = 2,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_2_config = {
    .super = {.name = "max_pooling1d_2"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};

const nnom_flatten_config_t flatten_config = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data[] = TENSOR_DENSE_KERNEL_0;

const nnom_shape_data_t tensor_dense_kernel_0_dim[] = {32, 2};
const nnom_qformat_param_t tensor_dense_kernel_0_dec[] = TENSOR_DENSE_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_dense_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_dense_kernel_0 = {
    .p_data = (void*)tensor_dense_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_dense_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_bias_0_data[] = TENSOR_DENSE_BIAS_0;

const nnom_shape_data_t tensor_dense_bias_0_dim[] = {2};
const nnom_qformat_param_t tensor_dense_bias_0_dec[] = TENSOR_DENSE_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_dense_bias_0_offset[] = {0};
const nnom_tensor_t tensor_dense_bias_0 = {
    .p_data = (void*)tensor_dense_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_dense_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_output_shift[] = DENSE_OUTPUT_RSHIFT;
const nnom_qformat_param_t dense_bias_shift[] = DENSE_BIAS_LSHIFT;
const nnom_dense_config_t dense_config = {
    .super = {.name = "dense"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_dense_bias_0,
    .output_shift = (nnom_qformat_param_t *)&dense_output_shift,
    .bias_shift = (nnom_qformat_param_t *)&dense_bias_shift
};
static int8_t nnom_output_data[2] = {0};

const nnom_shape_data_t tensor_output0_dim[] = {2};
const nnom_qformat_param_t tensor_output0_dec[] = {DENSE_OUTPUT_DEC};
const nnom_qformat_param_t tensor_output0_offset[] = {0};
const nnom_tensor_t tensor_output0 = {
    .p_data = (void*)nnom_output_data,
    .dim = (nnom_shape_data_t*)tensor_output0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_output0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_output0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output0_config = {
    .super = {.name = "output0"},
    .tensor = (nnom_tensor_t*)&tensor_output0
};
/* model version */
#define NNOM_MODEL_VERSION (10000*0 + 100*4 + 3)

/* nnom model */
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[13];

	check_model_version(NNOM_MODEL_VERSION);
	new_model(&model);

	layer[0] = input_s(&input_1_config);
	layer[1] = model.hook(conv2d_s(&conv1d_config), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(maxpool_s(&max_pooling1d_config), layer[2]);
	layer[4] = model.hook(conv2d_s(&conv1d_1_config), layer[3]);
	layer[5] = model.active(act_relu(), layer[4]);
	layer[6] = model.hook(maxpool_s(&max_pooling1d_1_config), layer[5]);
	layer[7] = model.hook(conv2d_s(&conv1d_2_config), layer[6]);
	layer[8] = model.active(act_relu(), layer[7]);
	layer[9] = model.hook(maxpool_s(&max_pooling1d_2_config), layer[8]);
	layer[10] = model.hook(flatten_s(&flatten_config), layer[9]);
	layer[11] = model.hook(dense_s(&dense_config), layer[10]);
	layer[12] = model.hook(output_s(&output0_config), layer[11]);
	model_compile(&model, layer[0], layer[12]);
	return &model;
}
