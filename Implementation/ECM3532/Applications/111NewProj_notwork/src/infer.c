#include "executor_public.h"

DECL_CONST_INT_ROM(pbuf_0_1,216);
DECL_CONST_INT_ROM(pbuf_0_2,8);
DECL_CONST_INT_ROM(pbuf_2_1,576);
DECL_CONST_INT_ROM(pbuf_2_2,8);
DECL_CONST_INT_ROM(pbuf_4_1,576);
DECL_CONST_INT_ROM(pbuf_4_2,8);
DECL_CONST_INT_ROM(pbuf_6_1,960);
DECL_CONST_INT_ROM(pbuf_6_2,10);

DECL_BUF_M3_PERSISTENT(pbuff0,6144);
DECL_BUF_M3_PERSISTENT(pbuff1,1536);
DECL_BUF_M3_PERSISTENT(pbuff2,1536);
DECL_BUF_M3_PERSISTENT(pbuff3,384);
DECL_BUF_M3_PERSISTENT(pbuff4,384);
DECL_BUF_M3_PERSISTENT(pbuff5,96);
DECL_BUF_M3_PERSISTENT(out0,10);
DECL_BUF_M3_PERSISTENT(inp0,2304);

DECL_BUF_M3_SCRATCH(pbuf_0_4,54);
DECL_BUF_M3_SCRATCH(pbuf_2_4,144);
DECL_BUF_M3_SCRATCH(pbuf_4_4,144);


void infer(q7_t *pIn0, q7_t *pOut0) { 

EXEC_ALLOC_MEM(inp0);
memcpy(GET_BUFF_ADDR(inp0), pIn0, NUM_BYTES(inp0));

EXEC_ALLOC_MEM(pbuff0);
conv2d_opt opt0 = { 
.in_rows = 24, .in_cols = 32, .in_depth = 3, .num_filt = 8, 
.filt_rows = 3, .filt_cols = 3, .row_pad = 1, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 24, .out_cols = 32,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_0_4);
EXEC_MAP_ROM(pbuf_0_1); EXEC_MAP_ROM(pbuf_0_2);
Exec_conv2d_q7(EXEC_HW_ID_DSP, &inp0, &pbuf_0_1, &pbuf_0_2, &pbuff0, &pbuf_0_4, &opt0);
EXEC_UNMAP_ROM(pbuf_0_1); EXEC_UNMAP_ROM(pbuf_0_2);

EXEC_FREE_MEM(inp0);
EXEC_ALLOC_MEM(pbuff1);
pool2d_opt opt1 = { .in_rows = 24, .in_cols = 32, .in_depth = 8, .filt_rows = 2, .filt_cols = 2, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 2, 
.out_lshift = 0, .out_rows = 12, .out_cols = 16,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff0, &pbuff1, &opt1);

EXEC_FREE_MEM(pbuff0);
EXEC_ALLOC_MEM(pbuff2);
conv2d_opt opt2 = { 
.in_rows = 12, .in_cols = 16, .in_depth = 8, .num_filt = 8, 
.filt_rows = 3, .filt_cols = 3, .row_pad = 1, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 12, .out_cols = 16,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_2_4);
EXEC_MAP_ROM(pbuf_2_1); EXEC_MAP_ROM(pbuf_2_2);
Exec_conv2d_q7(EXEC_HW_ID_DSP, &pbuff1, &pbuf_2_1, &pbuf_2_2, &pbuff2, &pbuf_2_4, &opt2);
EXEC_UNMAP_ROM(pbuf_2_1); EXEC_UNMAP_ROM(pbuf_2_2);

EXEC_FREE_MEM(pbuff1);
EXEC_ALLOC_MEM(pbuff3);
pool2d_opt opt3 = { .in_rows = 12, .in_cols = 16, .in_depth = 8, .filt_rows = 2, .filt_cols = 2, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 2, 
.out_lshift = 0, .out_rows = 6, .out_cols = 8,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff2, &pbuff3, &opt3);

EXEC_FREE_MEM(pbuff2);
EXEC_ALLOC_MEM(pbuff4);
conv2d_opt opt4 = { 
.in_rows = 6, .in_cols = 8, .in_depth = 8, .num_filt = 8, 
.filt_rows = 3, .filt_cols = 3, .row_pad = 1, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 6, .out_cols = 8,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_4_4);
EXEC_MAP_ROM(pbuf_4_1); EXEC_MAP_ROM(pbuf_4_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff3, &pbuf_4_1, &pbuf_4_2, &pbuff4, &pbuf_4_4, &opt4);
EXEC_UNMAP_ROM(pbuf_4_1); EXEC_UNMAP_ROM(pbuf_4_2);

EXEC_FREE_MEM(pbuff3);
EXEC_ALLOC_MEM(pbuff5);
pool2d_opt opt5 = { .in_rows = 6, .in_cols = 8, .in_depth = 8, .filt_rows = 2, .filt_cols = 2, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 2, 
.out_lshift = 0, .out_rows = 3, .out_cols = 4,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff4, &pbuff5, &opt5);

EXEC_FREE_MEM(pbuff4);
EXEC_ALLOC_MEM(out0);
fc_opt opt6 = { 
.filt_rows = 10, .filt_cols = 96, .bias_shift = 0, .out_shift = 7, .input_length = 96,
.act_min = 0, .act_max= 127};

EXEC_MAP_ROM(pbuf_6_1); EXEC_MAP_ROM(pbuf_6_2);
Exec_fully_connected_q7(EXEC_HW_ID_M3, &pbuff5, &pbuf_6_1, &pbuf_6_2, &out0, &opt6);
EXEC_UNMAP_ROM(pbuf_6_1); EXEC_UNMAP_ROM(pbuf_6_2);

EXEC_FREE_MEM(pbuff5);
WAIT_4_COMPLETION();
memcpy(pOut0, GET_BUFF_ADDR(out0), NUM_BYTES(out0));
EXEC_FREE_MEM(out0);
}
