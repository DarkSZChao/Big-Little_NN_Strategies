#include "executor_public.h"

DECL_CONST_INT_ROM(pbuf_1_1_S0,36);
DECL_CONST_INT_ROM(pbuf_1_2_S0,4);
DECL_CONST_INT_ROM(pbuf_5_1_S0,48);
DECL_CONST_INT_ROM(pbuf_5_2_S0,4);
DECL_CONST_INT_ROM(pbuf_9_1_S0,24);
DECL_CONST_INT_ROM(pbuf_9_2_S0,2);
DECL_CONST_INT_ROM(pbuf_13_1_S0,64);
DECL_CONST_INT_ROM(pbuf_13_2_S0,2);

DECL_BUF_M3_PERSISTENT(pbuff0_S0,384);
DECL_BUF_M3_PERSISTENT(pbuff1_S0,512);
DECL_BUF_M3_PERSISTENT(pbuff2_S0,512);
DECL_BUF_M3_PERSISTENT(pbuff3_S0,256);
DECL_BUF_M3_PERSISTENT(pbuff4_S0,256);
DECL_BUF_M3_PERSISTENT(pbuff5_S0,256);
DECL_BUF_M3_PERSISTENT(pbuff6_S0,256);
DECL_BUF_M3_PERSISTENT(pbuff7_S0,128);
DECL_BUF_M3_PERSISTENT(pbuff8_S0,128);
DECL_BUF_M3_PERSISTENT(pbuff9_S0,64);
DECL_BUF_M3_PERSISTENT(pbuff10_S0,64);
DECL_BUF_M3_PERSISTENT(pbuff11_S0,32);
DECL_BUF_M3_PERSISTENT(pbuff12_S0,32);
DECL_BUF_M3_PERSISTENT(out0_S0,2);
DECL_BUF_M3_PERSISTENT(inp0_S0,384);

DECL_BUF_M3_SCRATCH(pbuf_1_4_S0,18);
DECL_BUF_M3_SCRATCH(pbuf_5_4_S0,24);
DECL_BUF_M3_SCRATCH(pbuf_9_4_S0,24);


void infer_S0(q7_t *pIn0, q7_t *pOut0) { 

EXEC_ALLOC_MEM(inp0_S0);
memcpy(GET_BUFF_ADDR(inp0_S0), pIn0, NUM_BYTES(inp0_S0));

EXEC_ALLOC_MEM(pbuff0_S0);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff0_S0), GET_BUFF_ADDR(inp0_S0), NUM_BYTES(inp0_S0));
EXEC_FREE_MEM(inp0_S0);
EXEC_ALLOC_MEM(pbuff1_S0);
conv2d_opt opt1 = { 
.in_rows = 1, .in_cols = 128, .in_depth = 3, .num_filt = 4, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 1, .out_rshift = 7,
.out_rows = 1, .out_cols = 128,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_1_4_S0);
EXEC_MAP_ROM(pbuf_1_1_S0); EXEC_MAP_ROM(pbuf_1_2_S0);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff0_S0, &pbuf_1_1_S0, &pbuf_1_2_S0, &pbuff1_S0, &pbuf_1_4_S0, &opt1);
EXEC_UNMAP_ROM(pbuf_1_1_S0); EXEC_UNMAP_ROM(pbuf_1_2_S0);

EXEC_FREE_MEM(pbuff0_S0);
EXEC_ALLOC_MEM(pbuff2_S0);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff2_S0), GET_BUFF_ADDR(pbuff1_S0), NUM_BYTES(pbuff1_S0));
EXEC_FREE_MEM(pbuff1_S0);
EXEC_ALLOC_MEM(pbuff3_S0);
pool2d_opt opt3 = { .in_rows = 128, .in_cols = 1, .in_depth = 4, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 64, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff2_S0, &pbuff3_S0, &opt3);

EXEC_FREE_MEM(pbuff2_S0);
EXEC_ALLOC_MEM(pbuff4_S0);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff4_S0), GET_BUFF_ADDR(pbuff3_S0), NUM_BYTES(pbuff3_S0));
EXEC_FREE_MEM(pbuff3_S0);
EXEC_ALLOC_MEM(pbuff5_S0);
conv2d_opt opt5 = { 
.in_rows = 1, .in_cols = 64, .in_depth = 4, .num_filt = 4, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 6,
.out_rows = 1, .out_cols = 64,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_5_4_S0);
EXEC_MAP_ROM(pbuf_5_1_S0); EXEC_MAP_ROM(pbuf_5_2_S0);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff4_S0, &pbuf_5_1_S0, &pbuf_5_2_S0, &pbuff5_S0, &pbuf_5_4_S0, &opt5);
EXEC_UNMAP_ROM(pbuf_5_1_S0); EXEC_UNMAP_ROM(pbuf_5_2_S0);

EXEC_FREE_MEM(pbuff4_S0);
EXEC_ALLOC_MEM(pbuff6_S0);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff6_S0), GET_BUFF_ADDR(pbuff5_S0), NUM_BYTES(pbuff5_S0));
EXEC_FREE_MEM(pbuff5_S0);
EXEC_ALLOC_MEM(pbuff7_S0);
pool2d_opt opt7 = { .in_rows = 64, .in_cols = 1, .in_depth = 4, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 32, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff6_S0, &pbuff7_S0, &opt7);

EXEC_FREE_MEM(pbuff6_S0);
EXEC_ALLOC_MEM(pbuff8_S0);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff8_S0), GET_BUFF_ADDR(pbuff7_S0), NUM_BYTES(pbuff7_S0));
EXEC_FREE_MEM(pbuff7_S0);
EXEC_ALLOC_MEM(pbuff9_S0);
conv2d_opt opt9 = { 
.in_rows = 1, .in_cols = 32, .in_depth = 4, .num_filt = 2, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 6,
.out_rows = 1, .out_cols = 32,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_9_4_S0);
EXEC_MAP_ROM(pbuf_9_1_S0); EXEC_MAP_ROM(pbuf_9_2_S0);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff8_S0, &pbuf_9_1_S0, &pbuf_9_2_S0, &pbuff9_S0, &pbuf_9_4_S0, &opt9);
EXEC_UNMAP_ROM(pbuf_9_1_S0); EXEC_UNMAP_ROM(pbuf_9_2_S0);

EXEC_FREE_MEM(pbuff8_S0);
EXEC_ALLOC_MEM(pbuff10_S0);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff10_S0), GET_BUFF_ADDR(pbuff9_S0), NUM_BYTES(pbuff9_S0));
EXEC_FREE_MEM(pbuff9_S0);
EXEC_ALLOC_MEM(pbuff11_S0);
pool2d_opt opt11 = { .in_rows = 32, .in_cols = 1, .in_depth = 2, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 16, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff10_S0, &pbuff11_S0, &opt11);

EXEC_FREE_MEM(pbuff10_S0);
EXEC_ALLOC_MEM(pbuff12_S0);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff12_S0), GET_BUFF_ADDR(pbuff11_S0), NUM_BYTES(pbuff11_S0));
EXEC_FREE_MEM(pbuff11_S0);
EXEC_ALLOC_MEM(out0_S0);
fc_opt opt13 = { 
.filt_rows = 2, .filt_cols = 32, .bias_shift = 0, .out_shift = 7, .input_length = 32,
.act_min = 0, .act_max= 127};

EXEC_MAP_ROM(pbuf_13_1_S0); EXEC_MAP_ROM(pbuf_13_2_S0);
Exec_fully_connected_q7(EXEC_HW_ID_M3, &pbuff12_S0, &pbuf_13_1_S0, &pbuf_13_2_S0, &out0_S0, &opt13);
EXEC_UNMAP_ROM(pbuf_13_1_S0); EXEC_UNMAP_ROM(pbuf_13_2_S0);

EXEC_FREE_MEM(pbuff12_S0);
WAIT_4_COMPLETION();
memcpy(pOut0, GET_BUFF_ADDR(out0_S0), NUM_BYTES(out0_S0));
EXEC_FREE_MEM(out0_S0);
}
