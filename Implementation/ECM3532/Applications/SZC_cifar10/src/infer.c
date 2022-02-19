#include "executor_public.h"

DECL_CONST_INT_ROM(pbuf_0_1,540);
DECL_CONST_INT_ROM(pbuf_0_2,20);
DECL_CONST_INT_ROM(pbuf_2_1,3200);
DECL_CONST_INT_ROM(pbuf_2_2,40);
DECL_CONST_INT_ROM(pbuf_4_1,9600);
DECL_CONST_INT_ROM(pbuf_4_2,60);
DECL_CONST_INT_ROM(pbuf_5_1,32400);
DECL_CONST_INT_ROM(pbuf_5_2,60);
DECL_CONST_INT_ROM(pbuf_6_1,4800);
DECL_CONST_INT_ROM(pbuf_6_2,10);

DECL_BUF_M3_PERSISTENT(pbuff0,13200);
DECL_BUF_M3_PERSISTENT(pbuff1,3300);
DECL_BUF_M3_PERSISTENT(pbuff2,5600);
DECL_BUF_M3_PERSISTENT(pbuff3,1400);
DECL_BUF_M3_PERSISTENT(pbuff4,1440);
DECL_BUF_M3_PERSISTENT(pbuff5,480);
DECL_BUF_M3_PERSISTENT(pbuff6,10);
DECL_BUF_M3_PERSISTENT(out0,10);
DECL_BUF_M3_PERSISTENT(inp0,2304);

DECL_BUF_M3_SCRATCH(pbuf_0_4,54);
DECL_BUF_M3_SCRATCH(pbuf_2_4,160);
DECL_BUF_M3_SCRATCH(pbuf_4_4,320);
DECL_BUF_M3_SCRATCH(pbuf_5_4,1080);
DECL_BUF_M3_SCRATCH(pbuf_6_4,960);


void infer(q7_t *pIn0, q7_t *pOut0) { 

EXEC_ALLOC_MEM(inp0);
memcpy(GET_BUFF_ADDR(inp0), pIn0, NUM_BYTES(inp0));

EXEC_ALLOC_MEM(pbuff0);
conv2d_opt opt0 = { 
.in_rows = 24, .in_cols = 32, .in_depth = 3, .num_filt = 20, 
.filt_rows = 3, .filt_cols = 3, .row_pad = 0, .col_pad = 0, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 4, .out_rshift = 7,
.out_rows = 22, .out_cols = 30,.act_min = 0, .act_max = 124 };

EXEC_ALLOC_MEM(pbuf_0_4);
EXEC_MAP_ROM(pbuf_0_1); EXEC_MAP_ROM(pbuf_0_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &inp0, &pbuf_0_1, &pbuf_0_2, &pbuff0, &pbuf_0_4, &opt0);
EXEC_UNMAP_ROM(pbuf_0_1); EXEC_UNMAP_ROM(pbuf_0_2);

EXEC_FREE_MEM(inp0);
EXEC_ALLOC_MEM(pbuff1);
pool2d_opt opt1 = { .in_rows = 22, .in_cols = 30, .in_depth = 20, .filt_rows = 2, .filt_cols = 2, .row_pad = 0,
.col_pad = 0, .row_stride = 2, .col_stride = 2, .out_lshift = 0, .out_rows = 11, .out_cols = 15,.act_min = 0, .act_max = 124};

Exec_avepool2d_q7(EXEC_HW_ID_M3, &pbuff0, &pbuff1, &opt1);

EXEC_FREE_MEM(pbuff0);
EXEC_ALLOC_MEM(pbuff2);
conv2d_opt opt2 = { 
.in_rows = 11, .in_cols = 15, .in_depth = 20, .num_filt = 40, 
.filt_rows = 2, .filt_cols = 2, .row_pad = 0, .col_pad = 0, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 1, .out_rshift = 4,
.out_rows = 10, .out_cols = 14,.act_min = 0, .act_max = 119 };

EXEC_ALLOC_MEM(pbuf_2_4);
EXEC_MAP_ROM(pbuf_2_1); EXEC_MAP_ROM(pbuf_2_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff1, &pbuf_2_1, &pbuf_2_2, &pbuff2, &pbuf_2_4, &opt2);
EXEC_UNMAP_ROM(pbuf_2_1); EXEC_UNMAP_ROM(pbuf_2_2);

EXEC_FREE_MEM(pbuff1);
EXEC_ALLOC_MEM(pbuff3);
pool2d_opt opt3 = { .in_rows = 10, .in_cols = 14, .in_depth = 40, .filt_rows = 2, .filt_cols = 2, .row_pad = 0,
.col_pad = 0, .row_stride = 2, .col_stride = 2, .out_lshift = 0, .out_rows = 5, .out_cols = 7,.act_min = 0, .act_max = 119};

Exec_avepool2d_q7(EXEC_HW_ID_M3, &pbuff2, &pbuff3, &opt3);

EXEC_FREE_MEM(pbuff2);
EXEC_ALLOC_MEM(pbuff4);
conv2d_opt opt4 = { 
.in_rows = 5, .in_cols = 7, .in_depth = 40, .num_filt = 60, 
.filt_rows = 2, .filt_cols = 2, .row_pad = 0, .col_pad = 0, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 4, .out_rshift = 6,
.out_rows = 4, .out_cols = 6,.act_min = 0, .act_max = 94 };

EXEC_ALLOC_MEM(pbuf_4_4);
EXEC_MAP_ROM(pbuf_4_1); EXEC_MAP_ROM(pbuf_4_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff3, &pbuf_4_1, &pbuf_4_2, &pbuff4, &pbuf_4_4, &opt4);
EXEC_UNMAP_ROM(pbuf_4_1); EXEC_UNMAP_ROM(pbuf_4_2);

EXEC_FREE_MEM(pbuff3);
EXEC_ALLOC_MEM(pbuff5);
conv2d_opt opt5 = { 
.in_rows = 4, .in_cols = 6, .in_depth = 60, .num_filt = 60, 
.filt_rows = 3, .filt_cols = 3, .row_pad = 0, .col_pad = 0, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 4, .out_rshift = 7,
.out_rows = 2, .out_cols = 4,.act_min = 0, .act_max = 78 };

EXEC_ALLOC_MEM(pbuf_5_4);
EXEC_MAP_ROM(pbuf_5_1); EXEC_MAP_ROM(pbuf_5_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff4, &pbuf_5_1, &pbuf_5_2, &pbuff5, &pbuf_5_4, &opt5);
EXEC_UNMAP_ROM(pbuf_5_1); EXEC_UNMAP_ROM(pbuf_5_2);

EXEC_FREE_MEM(pbuff4);
EXEC_ALLOC_MEM(pbuff6);
conv2d_opt opt6 = { 
.in_rows = 2, .in_cols = 4, .in_depth = 60, .num_filt = 10, 
.filt_rows = 2, .filt_cols = 4, .row_pad = 0, .col_pad = 0, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 3, .out_rshift = 9,
.out_rows = 1, .out_cols = 1,.act_min = 0, .act_max = 71 };

EXEC_ALLOC_MEM(pbuf_6_4);
EXEC_MAP_ROM(pbuf_6_1); EXEC_MAP_ROM(pbuf_6_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff5, &pbuf_6_1, &pbuf_6_2, &pbuff6, &pbuf_6_4, &opt6);
EXEC_UNMAP_ROM(pbuf_6_1); EXEC_UNMAP_ROM(pbuf_6_2);

EXEC_FREE_MEM(pbuff5);
WAIT_4_COMPLETION();
EXEC_ALLOC_MEM(out0);
memcpy(GET_BUFF_ADDR(out0), GET_BUFF_ADDR(pbuff6), NUM_BYTES(pbuff6));
EXEC_FREE_MEM(pbuff6);
WAIT_4_COMPLETION();
memcpy(pOut0, GET_BUFF_ADDR(out0), NUM_BYTES(out0));
EXEC_FREE_MEM(out0);
}
