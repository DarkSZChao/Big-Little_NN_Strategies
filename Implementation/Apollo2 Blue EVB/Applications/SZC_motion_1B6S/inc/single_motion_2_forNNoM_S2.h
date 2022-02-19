#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0_S2 {46, -46, 2, 57, -4, 57, 22, 50, 30, -10, -48, -38, 20, -24, 9, 26, -47, 27, -44, -69, 12, -49, -57, 77, -46, 10, -19, -60, 109, 79, -35, -42, -16, 11, 49, 57}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS_S2 {7}

#define TENSOR_CONV1D_BIAS_0_S2 {62, -67, 52, -13}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS_S2 {6}

#define CONV1D_BIAS_LSHIFT_S2 {1}

#define CONV1D_OUTPUT_RSHIFT_S2 {8}

#define TENSOR_CONV1D_1_KERNEL_0_S2 {21, -36, 53, -35, 36, -11, 29, -14, 5, -17, -9, 4, -33, 7, 40, -29, 10, -7, 43, -15, 10, -63, -44, 13, 34, -9, -19, 6, 20, 23, 67, -33, -44, 51, -62, -56, 6, -7, 79, -90, 24, 55, 43, -6, 52, 30, -46, 28}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S2 {7}

#define TENSOR_CONV1D_1_BIAS_0_S2 {66, -1, 45, -50}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S2 {6}

#define CONV1D_1_BIAS_LSHIFT_S2 {0}

#define CONV1D_1_OUTPUT_RSHIFT_S2 {7}

#define TENSOR_CONV1D_2_KERNEL_0_S2 {-109, 54, -28, -14, 6, -54, -76, -37, -33, 40, 43, -13, -61, 64, -47, -77, -69, -82, 14, 10, -23, 3, -42, 41}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S2 {7}

#define TENSOR_CONV1D_2_BIAS_0_S2 {-32, -83}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S2 {6}

#define CONV1D_2_BIAS_LSHIFT_S2 {0}

#define CONV1D_2_OUTPUT_RSHIFT_S2 {7}

#define TENSOR_DENSE_KERNEL_0_S2 {-55, -14, -5, -25, -29, -10, 22, -35, -14, -40, -15, -26, -64, -54, 16, -71, -37, -3, 19, -46, -33, -14, -26, -15, 38, -67, -28, -58, -21, -30, -23, 14, 13, -11, 34, 20, 31, 23, -10, 8, -29, 17, -38, 42, -2, 6, -21, -3, -16, 62, 43, 30, -15, 29, 19, 50, -49, -19, -21, 7, -38, 11, 18, 65}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS_S2 {7}

#define TENSOR_DENSE_BIAS_0_S2 {103, -103}

#define TENSOR_DENSE_BIAS_0_DEC_BITS_S2 {5}

#define DENSE_BIAS_LSHIFT_S2 {1}

#define DENSE_OUTPUT_RSHIFT_S2 {5}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC_S2 0
#define INPUT_1_OUTPUT_OFFSET_S2 0
#define CONV1D_OUTPUT_DEC_S2 -1
#define CONV1D_OUTPUT_OFFSET_S2 0
#define RE_LU_OUTPUT_DEC_S2 -1
#define RE_LU_OUTPUT_OFFSET_S2 0
#define MAX_POOLING1D_OUTPUT_DEC_S2 -1
#define MAX_POOLING1D_OUTPUT_OFFSET_S2 0
#define CONV1D_1_OUTPUT_DEC_S2 -1
#define CONV1D_1_OUTPUT_OFFSET_S2 0
#define RE_LU_1_OUTPUT_DEC_S2 -1
#define RE_LU_1_OUTPUT_OFFSET_S2 0
#define MAX_POOLING1D_1_OUTPUT_DEC_S2 -1
#define MAX_POOLING1D_1_OUTPUT_OFFSET_S2 0
#define CONV1D_2_OUTPUT_DEC_S2 -1
#define CONV1D_2_OUTPUT_OFFSET_S2 0
#define RE_LU_2_OUTPUT_DEC_S2 -1
#define RE_LU_2_OUTPUT_OFFSET_S2 0
#define MAX_POOLING1D_2_OUTPUT_DEC_S2 -1
#define MAX_POOLING1D_2_OUTPUT_OFFSET_S2 0
#define FLATTEN_OUTPUT_DEC_S2 -1
#define FLATTEN_OUTPUT_OFFSET_S2 0
#define DENSE_OUTPUT_DEC_S2 1
#define DENSE_OUTPUT_OFFSET_S2 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data_S2[384] = {0};

const nnom_shape_data_t tensor_input_1_0_dim_S2[] = {128, 3};
const nnom_qformat_param_t tensor_input_1_0_dec_S2[] = {0};
const nnom_qformat_param_t tensor_input_1_0_offset_S2[] = {0};
const nnom_tensor_t tensor_input_1_0_S2 = {
    .p_data = (void*)nnom_input_data_S2,
    .dim = (nnom_shape_data_t*)tensor_input_1_0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config_S2 = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1_0_S2
};
const int8_t tensor_conv1d_kernel_0_data_S2[] = TENSOR_CONV1D_KERNEL_0_S2;

const nnom_shape_data_t tensor_conv1d_kernel_0_dim_S2[] = {3, 3, 4};
const nnom_qformat_param_t tensor_conv1d_kernel_0_dec_S2[] = TENSOR_CONV1D_KERNEL_0_DEC_BITS_S2;
const nnom_qformat_param_t tensor_conv1d_kernel_0_offset_S2[] = {0};
const nnom_tensor_t tensor_conv1d_kernel_0_S2 = {
    .p_data = (void*)tensor_conv1d_kernel_0_data_S2,
    .dim = (nnom_shape_data_t*)tensor_conv1d_kernel_0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_bias_0_data_S2[] = TENSOR_CONV1D_BIAS_0_S2;

const nnom_shape_data_t tensor_conv1d_bias_0_dim_S2[] = {4};
const nnom_qformat_param_t tensor_conv1d_bias_0_dec_S2[] = TENSOR_CONV1D_BIAS_0_DEC_BITS_S2;
const nnom_qformat_param_t tensor_conv1d_bias_0_offset_S2[] = {0};
const nnom_tensor_t tensor_conv1d_bias_0_S2 = {
    .p_data = (void*)tensor_conv1d_bias_0_data_S2,
    .dim = (nnom_shape_data_t*)tensor_conv1d_bias_0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_bias_0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_bias_0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_output_shift_S2[] = CONV1D_OUTPUT_RSHIFT_S2;
const nnom_qformat_param_t conv1d_bias_shift_S2[] = CONV1D_BIAS_LSHIFT_S2;
const nnom_conv2d_config_t conv1d_config_S2 = {
    .super = {.name = "conv1d"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_kernel_0_S2,
    .bias = (nnom_tensor_t*)&tensor_conv1d_bias_0_S2,
    .output_shift = (nnom_qformat_param_t *)&conv1d_output_shift_S2, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_bias_shift_S2, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_config_S2 = {
    .super = {.name = "max_pooling1d"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_1_kernel_0_data_S2[] = TENSOR_CONV1D_1_KERNEL_0_S2;

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim_S2[] = {3, 4, 4};
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_dec_S2[] = TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S2;
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_offset_S2[] = {0};
const nnom_tensor_t tensor_conv1d_1_kernel_0_S2 = {
    .p_data = (void*)tensor_conv1d_1_kernel_0_data_S2,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_kernel_0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_1_bias_0_data_S2[] = TENSOR_CONV1D_1_BIAS_0_S2;

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim_S2[] = {4};
const nnom_qformat_param_t tensor_conv1d_1_bias_0_dec_S2[] = TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S2;
const nnom_qformat_param_t tensor_conv1d_1_bias_0_offset_S2[] = {0};
const nnom_tensor_t tensor_conv1d_1_bias_0_S2 = {
    .p_data = (void*)tensor_conv1d_1_bias_0_data_S2,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_bias_0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_1_output_shift_S2[] = CONV1D_1_OUTPUT_RSHIFT_S2;
const nnom_qformat_param_t conv1d_1_bias_shift_S2[] = CONV1D_1_BIAS_LSHIFT_S2;
const nnom_conv2d_config_t conv1d_1_config_S2 = {
    .super = {.name = "conv1d_1"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_1_kernel_0_S2,
    .bias = (nnom_tensor_t*)&tensor_conv1d_1_bias_0_S2,
    .output_shift = (nnom_qformat_param_t *)&conv1d_1_output_shift_S2, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_1_bias_shift_S2, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_1_config_S2 = {
    .super = {.name = "max_pooling1d_1"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_2_kernel_0_data_S2[] = TENSOR_CONV1D_2_KERNEL_0_S2;

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim_S2[] = {3, 4, 2};
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_dec_S2[] = TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S2;
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_offset_S2[] = {0};
const nnom_tensor_t tensor_conv1d_2_kernel_0_S2 = {
    .p_data = (void*)tensor_conv1d_2_kernel_0_data_S2,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_kernel_0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_2_bias_0_data_S2[] = TENSOR_CONV1D_2_BIAS_0_S2;

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim_S2[] = {2};
const nnom_qformat_param_t tensor_conv1d_2_bias_0_dec_S2[] = TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S2;
const nnom_qformat_param_t tensor_conv1d_2_bias_0_offset_S2[] = {0};
const nnom_tensor_t tensor_conv1d_2_bias_0_S2 = {
    .p_data = (void*)tensor_conv1d_2_bias_0_data_S2,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_bias_0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_2_output_shift_S2[] = CONV1D_2_OUTPUT_RSHIFT_S2;
const nnom_qformat_param_t conv1d_2_bias_shift_S2[] = CONV1D_2_BIAS_LSHIFT_S2;
const nnom_conv2d_config_t conv1d_2_config_S2 = {
    .super = {.name = "conv1d_2"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_2_kernel_0_S2,
    .bias = (nnom_tensor_t*)&tensor_conv1d_2_bias_0_S2,
    .output_shift = (nnom_qformat_param_t *)&conv1d_2_output_shift_S2, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_2_bias_shift_S2, 
    .filter_size = 2,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_2_config_S2 = {
    .super = {.name = "max_pooling1d_2"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};

const nnom_flatten_config_t flatten_config_S2 = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data_S2[] = TENSOR_DENSE_KERNEL_0_S2;

const nnom_shape_data_t tensor_dense_kernel_0_dim_S2[] = {32, 2};
const nnom_qformat_param_t tensor_dense_kernel_0_dec_S2[] = TENSOR_DENSE_KERNEL_0_DEC_BITS_S2;
const nnom_qformat_param_t tensor_dense_kernel_0_offset_S2[] = {0};
const nnom_tensor_t tensor_dense_kernel_0_S2 = {
    .p_data = (void*)tensor_dense_kernel_0_data_S2,
    .dim = (nnom_shape_data_t*)tensor_dense_kernel_0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_kernel_0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_kernel_0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_bias_0_data_S2[] = TENSOR_DENSE_BIAS_0_S2;

const nnom_shape_data_t tensor_dense_bias_0_dim_S2[] = {2};
const nnom_qformat_param_t tensor_dense_bias_0_dec_S2[] = TENSOR_DENSE_BIAS_0_DEC_BITS_S2;
const nnom_qformat_param_t tensor_dense_bias_0_offset_S2[] = {0};
const nnom_tensor_t tensor_dense_bias_0_S2 = {
    .p_data = (void*)tensor_dense_bias_0_data_S2,
    .dim = (nnom_shape_data_t*)tensor_dense_bias_0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_bias_0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_bias_0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_output_shift_S2[] = DENSE_OUTPUT_RSHIFT_S2;
const nnom_qformat_param_t dense_bias_shift_S2[] = DENSE_BIAS_LSHIFT_S2;
const nnom_dense_config_t dense_config_S2 = {
    .super = {.name = "dense"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_kernel_0_S2,
    .bias = (nnom_tensor_t*)&tensor_dense_bias_0_S2,
    .output_shift = (nnom_qformat_param_t *)&dense_output_shift_S2,
    .bias_shift = (nnom_qformat_param_t *)&dense_bias_shift_S2
};
static int8_t nnom_output_data_S2[2] = {0};

const nnom_shape_data_t tensor_output0_dim_S2[] = {2};
const nnom_qformat_param_t tensor_output0_dec_S2[] = {DENSE_OUTPUT_DEC_S2};
const nnom_qformat_param_t tensor_output0_offset_S2[] = {0};
const nnom_tensor_t tensor_output0_S2 = {
    .p_data = (void*)nnom_output_data_S2,
    .dim = (nnom_shape_data_t*)tensor_output0_dim_S2,
    .q_dec = (nnom_qformat_param_t*)tensor_output0_dec_S2,
    .q_offset = (nnom_qformat_param_t*)tensor_output0_offset_S2,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output0_config_S2 = {
    .super = {.name = "output0"},
    .tensor = (nnom_tensor_t*)&tensor_output0_S2
};
/* model version */
#define NNOM_MODEL_VERSION_S2 (10000*0 + 100*4 + 3)

/* nnom model */
static nnom_model_t* nnom_model_create_S2(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[13];

	check_model_version(NNOM_MODEL_VERSION_S2);
	new_model(&model);

	layer[0] = input_s(&input_1_config_S2);
	layer[1] = model.hook(conv2d_s(&conv1d_config_S2), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(maxpool_s(&max_pooling1d_config_S2), layer[2]);
	layer[4] = model.hook(conv2d_s(&conv1d_1_config_S2), layer[3]);
	layer[5] = model.active(act_relu(), layer[4]);
	layer[6] = model.hook(maxpool_s(&max_pooling1d_1_config_S2), layer[5]);
	layer[7] = model.hook(conv2d_s(&conv1d_2_config_S2), layer[6]);
	layer[8] = model.active(act_relu(), layer[7]);
	layer[9] = model.hook(maxpool_s(&max_pooling1d_2_config_S2), layer[8]);
	layer[10] = model.hook(flatten_s(&flatten_config_S2), layer[9]);
	layer[11] = model.hook(dense_s(&dense_config_S2), layer[10]);
	layer[12] = model.hook(output_s(&output0_config_S2), layer[11]);
	model_compile(&model, layer[0], layer[12]);
	return &model;
}
