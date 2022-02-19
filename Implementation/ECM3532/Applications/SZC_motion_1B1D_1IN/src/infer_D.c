#include "executor_public.h"

DECL_CONST_INT_ROM(pbuf_1_1_D,24);
DECL_CONST_INT_ROM(pbuf_1_2_D,4);
DECL_CONST_INT_ROM(pbuf_5_1_D,48);
DECL_CONST_INT_ROM(pbuf_5_2_D,4);
DECL_CONST_INT_ROM(pbuf_9_1_D,24);
DECL_CONST_INT_ROM(pbuf_9_2_D,2);
DECL_CONST_INT_ROM(pbuf_13_1_D,192);
DECL_CONST_INT_ROM(pbuf_13_2_D,2);

DECL_BUF_M3_PERSISTENT(pbuff0_D,768);
DECL_BUF_M3_PERSISTENT(pbuff1_D,1536);
DECL_BUF_M3_PERSISTENT(pbuff2_D,1536);
DECL_BUF_M3_PERSISTENT(pbuff3_D,768);
DECL_BUF_M3_PERSISTENT(pbuff4_D,768);
DECL_BUF_M3_PERSISTENT(pbuff5_D,768);
DECL_BUF_M3_PERSISTENT(pbuff6_D,768);
DECL_BUF_M3_PERSISTENT(pbuff7_D,384);
DECL_BUF_M3_PERSISTENT(pbuff8_D,384);
DECL_BUF_M3_PERSISTENT(pbuff9_D,192);
DECL_BUF_M3_PERSISTENT(pbuff10_D,192);
DECL_BUF_M3_PERSISTENT(pbuff11_D,96);
DECL_BUF_M3_PERSISTENT(pbuff12_D,96);
DECL_BUF_M3_PERSISTENT(out0_D,2);
DECL_BUF_M3_PERSISTENT(inp0_D,768);

DECL_BUF_M3_SCRATCH(pbuf_1_4_D,12);
DECL_BUF_M3_SCRATCH(pbuf_5_4_D,24);
DECL_BUF_M3_SCRATCH(pbuf_9_4_D,24);


void infer_D(q7_t *pIn0, q7_t *pOut0) { 

EXEC_ALLOC_MEM(inp0_D);
memcpy(GET_BUFF_ADDR(inp0_D), pIn0, NUM_BYTES(inp0_D));

EXEC_ALLOC_MEM(pbuff0_D);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff0_D), GET_BUFF_ADDR(inp0_D), NUM_BYTES(inp0_D));
EXEC_FREE_MEM(inp0_D);
EXEC_ALLOC_MEM(pbuff1_D);
conv2d_opt opt1 = { 
.in_rows = 1, .in_cols = 384, .in_depth = 2, .num_filt = 4, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 8,
.out_rows = 1, .out_cols = 384,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_1_4_D);
EXEC_MAP_ROM(pbuf_1_1_D); EXEC_MAP_ROM(pbuf_1_2_D);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff0_D, &pbuf_1_1_D, &pbuf_1_2_D, &pbuff1_D, &pbuf_1_4_D, &opt1);
EXEC_UNMAP_ROM(pbuf_1_1_D); EXEC_UNMAP_ROM(pbuf_1_2_D);

EXEC_FREE_MEM(pbuff0_D);
EXEC_ALLOC_MEM(pbuff2_D);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff2_D), GET_BUFF_ADDR(pbuff1_D), NUM_BYTES(pbuff1_D));
EXEC_FREE_MEM(pbuff1_D);
EXEC_ALLOC_MEM(pbuff3_D);
pool2d_opt opt3 = { .in_rows = 384, .in_cols = 1, .in_depth = 4, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 192, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff2_D, &pbuff3_D, &opt3);

EXEC_FREE_MEM(pbuff2_D);
EXEC_ALLOC_MEM(pbuff4_D);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff4_D), GET_BUFF_ADDR(pbuff3_D), NUM_BYTES(pbuff3_D));
EXEC_FREE_MEM(pbuff3_D);
EXEC_ALLOC_MEM(pbuff5_D);
conv2d_opt opt5 = { 
.in_rows = 1, .in_cols = 192, .in_depth = 4, .num_filt = 4, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 192,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_5_4_D);
EXEC_MAP_ROM(pbuf_5_1_D); EXEC_MAP_ROM(pbuf_5_2_D);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff4_D, &pbuf_5_1_D, &pbuf_5_2_D, &pbuff5_D, &pbuf_5_4_D, &opt5);
EXEC_UNMAP_ROM(pbuf_5_1_D); EXEC_UNMAP_ROM(pbuf_5_2_D);

EXEC_FREE_MEM(pbuff4_D);
EXEC_ALLOC_MEM(pbuff6_D);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff6_D), GET_BUFF_ADDR(pbuff5_D), NUM_BYTES(pbuff5_D));
EXEC_FREE_MEM(pbuff5_D);
EXEC_ALLOC_MEM(pbuff7_D);
pool2d_opt opt7 = { .in_rows = 192, .in_cols = 1, .in_depth = 4, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 96, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff6_D, &pbuff7_D, &opt7);

EXEC_FREE_MEM(pbuff6_D);
EXEC_ALLOC_MEM(pbuff8_D);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff8_D), GET_BUFF_ADDR(pbuff7_D), NUM_BYTES(pbuff7_D));
EXEC_FREE_MEM(pbuff7_D);
EXEC_ALLOC_MEM(pbuff9_D);
conv2d_opt opt9 = { 
.in_rows = 1, .in_cols = 96, .in_depth = 4, .num_filt = 2, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 96,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_9_4_D);
EXEC_MAP_ROM(pbuf_9_1_D); EXEC_MAP_ROM(pbuf_9_2_D);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff8_D, &pbuf_9_1_D, &pbuf_9_2_D, &pbuff9_D, &pbuf_9_4_D, &opt9);
EXEC_UNMAP_ROM(pbuf_9_1_D); EXEC_UNMAP_ROM(pbuf_9_2_D);

EXEC_FREE_MEM(pbuff8_D);
EXEC_ALLOC_MEM(pbuff10_D);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff10_D), GET_BUFF_ADDR(pbuff9_D), NUM_BYTES(pbuff9_D));
EXEC_FREE_MEM(pbuff9_D);
EXEC_ALLOC_MEM(pbuff11_D);
pool2d_opt opt11 = { .in_rows = 96, .in_cols = 1, .in_depth = 2, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 48, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff10_D, &pbuff11_D, &opt11);

EXEC_FREE_MEM(pbuff10_D);
EXEC_ALLOC_MEM(pbuff12_D);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff12_D), GET_BUFF_ADDR(pbuff11_D), NUM_BYTES(pbuff11_D));
EXEC_FREE_MEM(pbuff11_D);
EXEC_ALLOC_MEM(out0_D);
fc_opt opt13 = { 
.filt_rows = 2, .filt_cols = 96, .bias_shift = 0, .out_shift = 7, .input_length = 96,
.act_min = 0, .act_max= 127};

EXEC_MAP_ROM(pbuf_13_1_D); EXEC_MAP_ROM(pbuf_13_2_D);
Exec_fully_connected_q7(EXEC_HW_ID_M3, &pbuff12_D, &pbuf_13_1_D, &pbuf_13_2_D, &out0_D, &opt13);
EXEC_UNMAP_ROM(pbuf_13_1_D); EXEC_UNMAP_ROM(pbuf_13_2_D);

EXEC_FREE_MEM(pbuff12_D);
WAIT_4_COMPLETION();
memcpy(pOut0, GET_BUFF_ADDR(out0_D), NUM_BYTES(out0_D));
EXEC_FREE_MEM(out0_D);
}
