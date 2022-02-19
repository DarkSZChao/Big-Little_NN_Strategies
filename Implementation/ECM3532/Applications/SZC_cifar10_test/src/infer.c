#include "executor_public.h"

DECL_CONST_INT_ROM(pbuf_0_1,864);
DECL_CONST_INT_ROM(pbuf_0_2,32);
DECL_CONST_INT_ROM(pbuf_2_1,18432);
DECL_CONST_INT_ROM(pbuf_2_2,64);
DECL_CONST_INT_ROM(pbuf_4_1,36864);
DECL_CONST_INT_ROM(pbuf_4_2,64);
DECL_CONST_INT_ROM(pbuf_6_1,65536);
DECL_CONST_INT_ROM(pbuf_6_2,64);
DECL_CONST_INT_ROM(pbuf_7_1,640);
DECL_CONST_INT_ROM(pbuf_7_2,10);

DECL_BUF_M3_PERSISTENT(pbuff0,32768);
DECL_BUF_M3_PERSISTENT(pbuff1,8192);
DECL_BUF_M3_PERSISTENT(pbuff2,16384);
DECL_BUF_M3_PERSISTENT(pbuff3,4096);
DECL_BUF_M3_PERSISTENT(pbuff4,4096);
DECL_BUF_M3_PERSISTENT(pbuff5,1024);
DECL_BUF_M3_PERSISTENT(pbuff6,64);
DECL_BUF_M3_PERSISTENT(out0,10);
DECL_BUF_M3_PERSISTENT(inp0,3072);

DECL_BUF_M3_SCRATCH(pbuf_0_4,54);
DECL_BUF_M3_SCRATCH(pbuf_2_4,576);
DECL_BUF_M3_SCRATCH(pbuf_4_4,1152);


void infer(q7_t *pIn0, q7_t *pOut0) { 

EXEC_ALLOC_MEM(inp0);
memcpy(GET_BUFF_ADDR(inp0), pIn0, NUM_BYTES(inp0));

EXEC_ALLOC_MEM(pbuff0);
conv2d_opt opt0 = { 
.in_rows = 32, .in_cols = 32, .in_depth = 3, .num_filt = 32, 
.filt_rows = 3, .filt_cols = 3, .row_pad = 1, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 32, .out_cols = 32,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_0_4);
EXEC_MAP_ROM(pbuf_0_1); EXEC_MAP_ROM(pbuf_0_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &inp0, &pbuf_0_1, &pbuf_0_2, &pbuff0, &pbuf_0_4, &opt0);
EXEC_UNMAP_ROM(pbuf_0_1); EXEC_UNMAP_ROM(pbuf_0_2);

EXEC_FREE_MEM(inp0);
EXEC_ALLOC_MEM(pbuff1);
pool2d_opt opt1 = { .in_rows = 32, .in_cols = 32, .in_depth = 32, .filt_rows = 2, .filt_cols = 2, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 2, 
.out_lshift = 0, .out_rows = 16, .out_cols = 16,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff0, &pbuff1, &opt1);

EXEC_FREE_MEM(pbuff0);
EXEC_ALLOC_MEM(pbuff2);
conv2d_opt opt2 = { 
.in_rows = 16, .in_cols = 16, .in_depth = 32, .num_filt = 64, 
.filt_rows = 3, .filt_cols = 3, .row_pad = 1, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 6,
.out_rows = 16, .out_cols = 16,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_2_4);
EXEC_MAP_ROM(pbuf_2_1); EXEC_MAP_ROM(pbuf_2_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff1, &pbuf_2_1, &pbuf_2_2, &pbuff2, &pbuf_2_4, &opt2);
EXEC_UNMAP_ROM(pbuf_2_1); EXEC_UNMAP_ROM(pbuf_2_2);

EXEC_FREE_MEM(pbuff1);
EXEC_ALLOC_MEM(pbuff3);
pool2d_opt opt3 = { .in_rows = 16, .in_cols = 16, .in_depth = 64, .filt_rows = 2, .filt_cols = 2, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 2, 
.out_lshift = 0, .out_rows = 8, .out_cols = 8,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff2, &pbuff3, &opt3);

EXEC_FREE_MEM(pbuff2);
EXEC_ALLOC_MEM(pbuff4);
conv2d_opt opt4 = { 
.in_rows = 8, .in_cols = 8, .in_depth = 64, .num_filt = 64, 
.filt_rows = 3, .filt_cols = 3, .row_pad = 1, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 1, .out_rshift = 7,
.out_rows = 8, .out_cols = 8,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_4_4);
EXEC_MAP_ROM(pbuf_4_1); EXEC_MAP_ROM(pbuf_4_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff3, &pbuf_4_1, &pbuf_4_2, &pbuff4, &pbuf_4_4, &opt4);
EXEC_UNMAP_ROM(pbuf_4_1); EXEC_UNMAP_ROM(pbuf_4_2);

EXEC_FREE_MEM(pbuff3);
EXEC_ALLOC_MEM(pbuff5);
pool2d_opt opt5 = { .in_rows = 8, .in_cols = 8, .in_depth = 64, .filt_rows = 2, .filt_cols = 2, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 2, 
.out_lshift = 0, .out_rows = 4, .out_cols = 4,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff4, &pbuff5, &opt5);

EXEC_FREE_MEM(pbuff4);
EXEC_ALLOC_MEM(pbuff6);
fc_opt opt6 = { 
.filt_rows = 64, .filt_cols = 1024, .bias_shift = 0, .out_shift = 7, .input_length = 1024,
.act_min = 0, .act_max= 127};

EXEC_MAP_ROM(pbuf_6_1); EXEC_MAP_ROM(pbuf_6_2);
Exec_fully_connected_q7(EXEC_HW_ID_M3, &pbuff5, &pbuf_6_1, &pbuf_6_2, &pbuff6, &opt6);
EXEC_UNMAP_ROM(pbuf_6_1); EXEC_UNMAP_ROM(pbuf_6_2);

EXEC_FREE_MEM(pbuff5);
EXEC_ALLOC_MEM(out0);
fc_opt opt7 = { 
.filt_rows = 10, .filt_cols = 64, .bias_shift = 0, .out_shift = 7, .input_length = 64,
.act_min = 0, .act_max= 127};

EXEC_MAP_ROM(pbuf_7_1); EXEC_MAP_ROM(pbuf_7_2);
Exec_fully_connected_q7(EXEC_HW_ID_M3, &pbuff6, &pbuf_7_1, &pbuf_7_2, &out0, &opt7);
EXEC_UNMAP_ROM(pbuf_7_1); EXEC_UNMAP_ROM(pbuf_7_2);

EXEC_FREE_MEM(pbuff6);
WAIT_4_COMPLETION();
memcpy(pOut0, GET_BUFF_ADDR(out0), NUM_BYTES(out0));
EXEC_FREE_MEM(out0);
}
