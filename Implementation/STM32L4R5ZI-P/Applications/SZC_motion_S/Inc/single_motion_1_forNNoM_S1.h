#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0_S1 {49, -68, -42, -43, -8, -34, 9, -72, 45, -22, 60, 13, -33, -28, 24, -6, -70, -28, -45, -11, -42, -12, -32, 23, 44, 43, -30, 33, 21, -13, -5, -31, -1, 11, 29, 14}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS_S1 {7}

#define TENSOR_CONV1D_BIAS_0_S1 {-65, 76, -64, 18}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS_S1 {6}

#define CONV1D_BIAS_LSHIFT_S1 {1}

#define CONV1D_OUTPUT_RSHIFT_S1 {8}

#define TENSOR_CONV1D_1_KERNEL_0_S1 {30, 9, 117, -23, -49, -24, 26, 41, 29, 51, 60, -4, -29, -8, 12, 24, -23, 15, 16, 32, 36, -85, 14, -4, 18, -6, -44, 67, -51, -3, 19, -9, -22, -4, 24, 26, 23, 4, 63, -3, -61, -17, 29, -37, 28, -16, -48, 23}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S1 {7}

#define TENSOR_CONV1D_1_BIAS_0_S1 {25, -68, 33, -43}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S1 {6}

#define CONV1D_1_BIAS_LSHIFT_S1 {0}

#define CONV1D_1_OUTPUT_RSHIFT_S1 {6}

#define TENSOR_CONV1D_2_KERNEL_0_S1 {42, -29, 26, 33, -35, -94, 67, 1, -65, -16, -10, 56, 21, 11, 7, 8, -63, 8, 20, 37, 23, 68, -48, -59}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S1 {7}

#define TENSOR_CONV1D_2_BIAS_0_S1 {-38, 19}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S1 {7}

#define CONV1D_2_BIAS_LSHIFT_S1 {0}

#define CONV1D_2_OUTPUT_RSHIFT_S1 {6}

#define TENSOR_DENSE_KERNEL_0_S1 {21, 9, 41, -4, 9, -16, 44, 8, 26, -26, 51, -45, 15, -10, 35, -35, -5, -18, 27, -34, 14, 6, 82, -46, 82, -31, 5, 15, 58, -35, -14, 28, -10, 24, -17, 27, -58, 18, -31, 56, -70, 36, 2, 1, -84, 40, -38, 35, -76, 26, -66, 5, -60, 37, -5, 0, -28, 26, -37, 44, -19, 8, -9, -43}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS_S1 {7}

#define TENSOR_DENSE_BIAS_0_S1 {79, -79}

#define TENSOR_DENSE_BIAS_0_DEC_BITS_S1 {7}

#define DENSE_BIAS_LSHIFT_S1 {1}

#define DENSE_OUTPUT_RSHIFT_S1 {6}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC_S1 0
#define INPUT_1_OUTPUT_OFFSET_S1 0
#define CONV1D_OUTPUT_DEC_S1 -1
#define CONV1D_OUTPUT_OFFSET_S1 0
#define RE_LU_OUTPUT_DEC_S1 -1
#define RE_LU_OUTPUT_OFFSET_S1 0
#define MAX_POOLING1D_OUTPUT_DEC_S1 -1
#define MAX_POOLING1D_OUTPUT_OFFSET_S1 0
#define CONV1D_1_OUTPUT_DEC_S1 0
#define CONV1D_1_OUTPUT_OFFSET_S1 0
#define RE_LU_1_OUTPUT_DEC_S1 0
#define RE_LU_1_OUTPUT_OFFSET_S1 0
#define MAX_POOLING1D_1_OUTPUT_DEC_S1 0
#define MAX_POOLING1D_1_OUTPUT_OFFSET_S1 0
#define CONV1D_2_OUTPUT_DEC_S1 1
#define CONV1D_2_OUTPUT_OFFSET_S1 0
#define RE_LU_2_OUTPUT_DEC_S1 1
#define RE_LU_2_OUTPUT_OFFSET_S1 0
#define MAX_POOLING1D_2_OUTPUT_DEC_S1 1
#define MAX_POOLING1D_2_OUTPUT_OFFSET_S1 0
#define FLATTEN_OUTPUT_DEC_S1 1
#define FLATTEN_OUTPUT_OFFSET_S1 0
#define DENSE_OUTPUT_DEC_S1 2
#define DENSE_OUTPUT_OFFSET_S1 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data_S1[384] = {0};

const nnom_shape_data_t tensor_input_1_0_dim_S1[] = {128, 3};
const nnom_qformat_param_t tensor_input_1_0_dec_S1[] = {0};
const nnom_qformat_param_t tensor_input_1_0_offset_S1[] = {0};
const nnom_tensor_t tensor_input_1_0_S1 = {
    .p_data = (void*)nnom_input_data_S1,
    .dim = (nnom_shape_data_t*)tensor_input_1_0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config_S1 = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1_0_S1
};
const int8_t tensor_conv1d_kernel_0_data_S1[] = TENSOR_CONV1D_KERNEL_0_S1;

const nnom_shape_data_t tensor_conv1d_kernel_0_dim_S1[] = {3, 3, 4};
const nnom_qformat_param_t tensor_conv1d_kernel_0_dec_S1[] = TENSOR_CONV1D_KERNEL_0_DEC_BITS_S1;
const nnom_qformat_param_t tensor_conv1d_kernel_0_offset_S1[] = {0};
const nnom_tensor_t tensor_conv1d_kernel_0_S1 = {
    .p_data = (void*)tensor_conv1d_kernel_0_data_S1,
    .dim = (nnom_shape_data_t*)tensor_conv1d_kernel_0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_bias_0_data_S1[] = TENSOR_CONV1D_BIAS_0_S1;

const nnom_shape_data_t tensor_conv1d_bias_0_dim_S1[] = {4};
const nnom_qformat_param_t tensor_conv1d_bias_0_dec_S1[] = TENSOR_CONV1D_BIAS_0_DEC_BITS_S1;
const nnom_qformat_param_t tensor_conv1d_bias_0_offset_S1[] = {0};
const nnom_tensor_t tensor_conv1d_bias_0_S1 = {
    .p_data = (void*)tensor_conv1d_bias_0_data_S1,
    .dim = (nnom_shape_data_t*)tensor_conv1d_bias_0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_bias_0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_bias_0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_output_shift_S1[] = CONV1D_OUTPUT_RSHIFT_S1;
const nnom_qformat_param_t conv1d_bias_shift_S1[] = CONV1D_BIAS_LSHIFT_S1;
const nnom_conv2d_config_t conv1d_config_S1 = {
    .super = {.name = "conv1d"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_kernel_0_S1,
    .bias = (nnom_tensor_t*)&tensor_conv1d_bias_0_S1,
    .output_shift = (nnom_qformat_param_t *)&conv1d_output_shift_S1, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_bias_shift_S1, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_config_S1 = {
    .super = {.name = "max_pooling1d"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_1_kernel_0_data_S1[] = TENSOR_CONV1D_1_KERNEL_0_S1;

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim_S1[] = {3, 4, 4};
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_dec_S1[] = TENSOR_CONV1D_1_KERNEL_0_DEC_BITS_S1;
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_offset_S1[] = {0};
const nnom_tensor_t tensor_conv1d_1_kernel_0_S1 = {
    .p_data = (void*)tensor_conv1d_1_kernel_0_data_S1,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_kernel_0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_1_bias_0_data_S1[] = TENSOR_CONV1D_1_BIAS_0_S1;

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim_S1[] = {4};
const nnom_qformat_param_t tensor_conv1d_1_bias_0_dec_S1[] = TENSOR_CONV1D_1_BIAS_0_DEC_BITS_S1;
const nnom_qformat_param_t tensor_conv1d_1_bias_0_offset_S1[] = {0};
const nnom_tensor_t tensor_conv1d_1_bias_0_S1 = {
    .p_data = (void*)tensor_conv1d_1_bias_0_data_S1,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_bias_0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_1_output_shift_S1[] = CONV1D_1_OUTPUT_RSHIFT_S1;
const nnom_qformat_param_t conv1d_1_bias_shift_S1[] = CONV1D_1_BIAS_LSHIFT_S1;
const nnom_conv2d_config_t conv1d_1_config_S1 = {
    .super = {.name = "conv1d_1"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_1_kernel_0_S1,
    .bias = (nnom_tensor_t*)&tensor_conv1d_1_bias_0_S1,
    .output_shift = (nnom_qformat_param_t *)&conv1d_1_output_shift_S1, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_1_bias_shift_S1, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_1_config_S1 = {
    .super = {.name = "max_pooling1d_1"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_2_kernel_0_data_S1[] = TENSOR_CONV1D_2_KERNEL_0_S1;

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim_S1[] = {3, 4, 2};
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_dec_S1[] = TENSOR_CONV1D_2_KERNEL_0_DEC_BITS_S1;
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_offset_S1[] = {0};
const nnom_tensor_t tensor_conv1d_2_kernel_0_S1 = {
    .p_data = (void*)tensor_conv1d_2_kernel_0_data_S1,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_kernel_0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_2_bias_0_data_S1[] = TENSOR_CONV1D_2_BIAS_0_S1;

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim_S1[] = {2};
const nnom_qformat_param_t tensor_conv1d_2_bias_0_dec_S1[] = TENSOR_CONV1D_2_BIAS_0_DEC_BITS_S1;
const nnom_qformat_param_t tensor_conv1d_2_bias_0_offset_S1[] = {0};
const nnom_tensor_t tensor_conv1d_2_bias_0_S1 = {
    .p_data = (void*)tensor_conv1d_2_bias_0_data_S1,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_bias_0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_2_output_shift_S1[] = CONV1D_2_OUTPUT_RSHIFT_S1;
const nnom_qformat_param_t conv1d_2_bias_shift_S1[] = CONV1D_2_BIAS_LSHIFT_S1;
const nnom_conv2d_config_t conv1d_2_config_S1 = {
    .super = {.name = "conv1d_2"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_2_kernel_0_S1,
    .bias = (nnom_tensor_t*)&tensor_conv1d_2_bias_0_S1,
    .output_shift = (nnom_qformat_param_t *)&conv1d_2_output_shift_S1, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_2_bias_shift_S1, 
    .filter_size = 2,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_2_config_S1 = {
    .super = {.name = "max_pooling1d_2"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};

const nnom_flatten_config_t flatten_config_S1 = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data_S1[] = TENSOR_DENSE_KERNEL_0_S1;

const nnom_shape_data_t tensor_dense_kernel_0_dim_S1[] = {32, 2};
const nnom_qformat_param_t tensor_dense_kernel_0_dec_S1[] = TENSOR_DENSE_KERNEL_0_DEC_BITS_S1;
const nnom_qformat_param_t tensor_dense_kernel_0_offset_S1[] = {0};
const nnom_tensor_t tensor_dense_kernel_0_S1 = {
    .p_data = (void*)tensor_dense_kernel_0_data_S1,
    .dim = (nnom_shape_data_t*)tensor_dense_kernel_0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_kernel_0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_kernel_0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_bias_0_data_S1[] = TENSOR_DENSE_BIAS_0_S1;

const nnom_shape_data_t tensor_dense_bias_0_dim_S1[] = {2};
const nnom_qformat_param_t tensor_dense_bias_0_dec_S1[] = TENSOR_DENSE_BIAS_0_DEC_BITS_S1;
const nnom_qformat_param_t tensor_dense_bias_0_offset_S1[] = {0};
const nnom_tensor_t tensor_dense_bias_0_S1 = {
    .p_data = (void*)tensor_dense_bias_0_data_S1,
    .dim = (nnom_shape_data_t*)tensor_dense_bias_0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_bias_0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_bias_0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_output_shift_S1[] = DENSE_OUTPUT_RSHIFT_S1;
const nnom_qformat_param_t dense_bias_shift_S1[] = DENSE_BIAS_LSHIFT_S1;
const nnom_dense_config_t dense_config_S1 = {
    .super = {.name = "dense"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_kernel_0_S1,
    .bias = (nnom_tensor_t*)&tensor_dense_bias_0_S1,
    .output_shift = (nnom_qformat_param_t *)&dense_output_shift_S1,
    .bias_shift = (nnom_qformat_param_t *)&dense_bias_shift_S1
};
static int8_t nnom_output_data_S1[2] = {0};

const nnom_shape_data_t tensor_output0_dim_S1[] = {2};
const nnom_qformat_param_t tensor_output0_dec_S1[] = {DENSE_OUTPUT_DEC_S1};
const nnom_qformat_param_t tensor_output0_offset_S1[] = {0};
const nnom_tensor_t tensor_output0_S1 = {
    .p_data = (void*)nnom_output_data_S1,
    .dim = (nnom_shape_data_t*)tensor_output0_dim_S1,
    .q_dec = (nnom_qformat_param_t*)tensor_output0_dec_S1,
    .q_offset = (nnom_qformat_param_t*)tensor_output0_offset_S1,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output0_config_S1 = {
    .super = {.name = "output0"},
    .tensor = (nnom_tensor_t*)&tensor_output0_S1
};
/* model version */
#define NNOM_MODEL_VERSION_S1 (10000*0 + 100*4 + 3)

/* nnom model */
static nnom_model_t* nnom_model_create_S1(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[13];

	check_model_version(NNOM_MODEL_VERSION_S1);
	new_model(&model);

	layer[0] = input_s(&input_1_config_S1);
	layer[1] = model.hook(conv2d_s(&conv1d_config_S1), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(maxpool_s(&max_pooling1d_config_S1), layer[2]);
	layer[4] = model.hook(conv2d_s(&conv1d_1_config_S1), layer[3]);
	layer[5] = model.active(act_relu(), layer[4]);
	layer[6] = model.hook(maxpool_s(&max_pooling1d_1_config_S1), layer[5]);
	layer[7] = model.hook(conv2d_s(&conv1d_2_config_S1), layer[6]);
	layer[8] = model.active(act_relu(), layer[7]);
	layer[9] = model.hook(maxpool_s(&max_pooling1d_2_config_S1), layer[8]);
	layer[10] = model.hook(flatten_s(&flatten_config_S1), layer[9]);
	layer[11] = model.hook(dense_s(&dense_config_S1), layer[10]);
	layer[12] = model.hook(output_s(&output0_config_S1), layer[11]);
	model_compile(&model, layer[0], layer[12]);
	return &model;
}
