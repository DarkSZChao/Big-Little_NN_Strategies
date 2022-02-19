#include "executor_public.h"

DECL_CONST_INT_ROM(pbuf_1_1,108);
DECL_CONST_INT_ROM(pbuf_1_2,4);
DECL_CONST_INT_ROM(pbuf_5_1,96);
DECL_CONST_INT_ROM(pbuf_5_2,8);
DECL_CONST_INT_ROM(pbuf_9_1,384);
DECL_CONST_INT_ROM(pbuf_9_2,16);
DECL_CONST_INT_ROM(pbuf_13_1,1536);
DECL_CONST_INT_ROM(pbuf_13_2,32);
DECL_CONST_INT_ROM(pbuf_17_1,768);
DECL_CONST_INT_ROM(pbuf_17_2,8);
DECL_CONST_INT_ROM(pbuf_21_1,192);
DECL_CONST_INT_ROM(pbuf_21_2,6);

DECL_BUF_M3_PERSISTENT(pbuff0,1152);
DECL_BUF_M3_PERSISTENT(pbuff1,512);
DECL_BUF_M3_PERSISTENT(pbuff2,512);
DECL_BUF_M3_PERSISTENT(pbuff3,256);
DECL_BUF_M3_PERSISTENT(pbuff4,256);
DECL_BUF_M3_PERSISTENT(pbuff5,512);
DECL_BUF_M3_PERSISTENT(pbuff6,512);
DECL_BUF_M3_PERSISTENT(pbuff7,256);
DECL_BUF_M3_PERSISTENT(pbuff8,256);
DECL_BUF_M3_PERSISTENT(pbuff9,512);
DECL_BUF_M3_PERSISTENT(pbuff10,512);
DECL_BUF_M3_PERSISTENT(pbuff11,256);
DECL_BUF_M3_PERSISTENT(pbuff12,256);
DECL_BUF_M3_PERSISTENT(pbuff13,512);
DECL_BUF_M3_PERSISTENT(pbuff14,512);
DECL_BUF_M3_PERSISTENT(pbuff15,256);
DECL_BUF_M3_PERSISTENT(pbuff16,256);
DECL_BUF_M3_PERSISTENT(pbuff17,64);
DECL_BUF_M3_PERSISTENT(pbuff18,64);
DECL_BUF_M3_PERSISTENT(pbuff19,32);
DECL_BUF_M3_PERSISTENT(pbuff20,32);
DECL_BUF_M3_PERSISTENT(out0,6);
DECL_BUF_M3_PERSISTENT(inp0,1152);

DECL_BUF_M3_SCRATCH(pbuf_1_4,54);
DECL_BUF_M3_SCRATCH(pbuf_5_4,24);
DECL_BUF_M3_SCRATCH(pbuf_9_4,48);
DECL_BUF_M3_SCRATCH(pbuf_13_4,96);
DECL_BUF_M3_SCRATCH(pbuf_17_4,192);


void infer(q7_t *pIn0, q7_t *pOut0) { 

EXEC_ALLOC_MEM(inp0);
memcpy(GET_BUFF_ADDR(inp0), pIn0, NUM_BYTES(inp0));

EXEC_ALLOC_MEM(pbuff0);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff0), GET_BUFF_ADDR(inp0), NUM_BYTES(inp0));
EXEC_FREE_MEM(inp0);
EXEC_ALLOC_MEM(pbuff1);
conv2d_opt opt1 = { 
.in_rows = 1, .in_cols = 128, .in_depth = 9, .num_filt = 4, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 8,
.out_rows = 1, .out_cols = 128,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_1_4);
EXEC_MAP_ROM(pbuf_1_1); EXEC_MAP_ROM(pbuf_1_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff0, &pbuf_1_1, &pbuf_1_2, &pbuff1, &pbuf_1_4, &opt1);
EXEC_UNMAP_ROM(pbuf_1_1); EXEC_UNMAP_ROM(pbuf_1_2);

EXEC_FREE_MEM(pbuff0);
EXEC_ALLOC_MEM(pbuff2);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff2), GET_BUFF_ADDR(pbuff1), NUM_BYTES(pbuff1));
EXEC_FREE_MEM(pbuff1);
EXEC_ALLOC_MEM(pbuff3);
pool2d_opt opt3 = { .in_rows = 128, .in_cols = 1, .in_depth = 4, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 64, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff2, &pbuff3, &opt3);

EXEC_FREE_MEM(pbuff2);
EXEC_ALLOC_MEM(pbuff4);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff4), GET_BUFF_ADDR(pbuff3), NUM_BYTES(pbuff3));
EXEC_FREE_MEM(pbuff3);
EXEC_ALLOC_MEM(pbuff5);
conv2d_opt opt5 = { 
.in_rows = 1, .in_cols = 64, .in_depth = 4, .num_filt = 8, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 64,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_5_4);
EXEC_MAP_ROM(pbuf_5_1); EXEC_MAP_ROM(pbuf_5_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff4, &pbuf_5_1, &pbuf_5_2, &pbuff5, &pbuf_5_4, &opt5);
EXEC_UNMAP_ROM(pbuf_5_1); EXEC_UNMAP_ROM(pbuf_5_2);

EXEC_FREE_MEM(pbuff4);
EXEC_ALLOC_MEM(pbuff6);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff6), GET_BUFF_ADDR(pbuff5), NUM_BYTES(pbuff5));
EXEC_FREE_MEM(pbuff5);
EXEC_ALLOC_MEM(pbuff7);
pool2d_opt opt7 = { .in_rows = 64, .in_cols = 1, .in_depth = 8, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 32, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff6, &pbuff7, &opt7);

EXEC_FREE_MEM(pbuff6);
EXEC_ALLOC_MEM(pbuff8);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff8), GET_BUFF_ADDR(pbuff7), NUM_BYTES(pbuff7));
EXEC_FREE_MEM(pbuff7);
EXEC_ALLOC_MEM(pbuff9);
conv2d_opt opt9 = { 
.in_rows = 1, .in_cols = 32, .in_depth = 8, .num_filt = 16, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 32,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_9_4);
EXEC_MAP_ROM(pbuf_9_1); EXEC_MAP_ROM(pbuf_9_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff8, &pbuf_9_1, &pbuf_9_2, &pbuff9, &pbuf_9_4, &opt9);
EXEC_UNMAP_ROM(pbuf_9_1); EXEC_UNMAP_ROM(pbuf_9_2);

EXEC_FREE_MEM(pbuff8);
EXEC_ALLOC_MEM(pbuff10);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff10), GET_BUFF_ADDR(pbuff9), NUM_BYTES(pbuff9));
EXEC_FREE_MEM(pbuff9);
EXEC_ALLOC_MEM(pbuff11);
pool2d_opt opt11 = { .in_rows = 32, .in_cols = 1, .in_depth = 16, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 16, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff10, &pbuff11, &opt11);

EXEC_FREE_MEM(pbuff10);
EXEC_ALLOC_MEM(pbuff12);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff12), GET_BUFF_ADDR(pbuff11), NUM_BYTES(pbuff11));
EXEC_FREE_MEM(pbuff11);
EXEC_ALLOC_MEM(pbuff13);
conv2d_opt opt13 = { 
.in_rows = 1, .in_cols = 16, .in_depth = 16, .num_filt = 32, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 16,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_13_4);
EXEC_MAP_ROM(pbuf_13_1); EXEC_MAP_ROM(pbuf_13_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff12, &pbuf_13_1, &pbuf_13_2, &pbuff13, &pbuf_13_4, &opt13);
EXEC_UNMAP_ROM(pbuf_13_1); EXEC_UNMAP_ROM(pbuf_13_2);

EXEC_FREE_MEM(pbuff12);
EXEC_ALLOC_MEM(pbuff14);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff14), GET_BUFF_ADDR(pbuff13), NUM_BYTES(pbuff13));
EXEC_FREE_MEM(pbuff13);
EXEC_ALLOC_MEM(pbuff15);
pool2d_opt opt15 = { .in_rows = 16, .in_cols = 1, .in_depth = 32, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 8, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff14, &pbuff15, &opt15);

EXEC_FREE_MEM(pbuff14);
EXEC_ALLOC_MEM(pbuff16);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff16), GET_BUFF_ADDR(pbuff15), NUM_BYTES(pbuff15));
EXEC_FREE_MEM(pbuff15);
EXEC_ALLOC_MEM(pbuff17);
conv2d_opt opt17 = { 
.in_rows = 1, .in_cols = 8, .in_depth = 32, .num_filt = 8, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 8,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_17_4);
EXEC_MAP_ROM(pbuf_17_1); EXEC_MAP_ROM(pbuf_17_2);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff16, &pbuf_17_1, &pbuf_17_2, &pbuff17, &pbuf_17_4, &opt17);
EXEC_UNMAP_ROM(pbuf_17_1); EXEC_UNMAP_ROM(pbuf_17_2);

EXEC_FREE_MEM(pbuff16);
EXEC_ALLOC_MEM(pbuff18);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff18), GET_BUFF_ADDR(pbuff17), NUM_BYTES(pbuff17));
EXEC_FREE_MEM(pbuff17);
EXEC_ALLOC_MEM(pbuff19);
pool2d_opt opt19 = { .in_rows = 8, .in_cols = 1, .in_depth = 8, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 4, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff18, &pbuff19, &opt19);

EXEC_FREE_MEM(pbuff18);
EXEC_ALLOC_MEM(pbuff20);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff20), GET_BUFF_ADDR(pbuff19), NUM_BYTES(pbuff19));
EXEC_FREE_MEM(pbuff19);
EXEC_ALLOC_MEM(out0);
fc_opt opt21 = { 
.filt_rows = 6, .filt_cols = 32, .bias_shift = 0, .out_shift = 6, .input_length = 32,
.act_min = 0, .act_max= 127};

EXEC_MAP_ROM(pbuf_21_1); EXEC_MAP_ROM(pbuf_21_2);
Exec_fully_connected_q7(EXEC_HW_ID_M3, &pbuff20, &pbuf_21_1, &pbuf_21_2, &out0, &opt21);
EXEC_UNMAP_ROM(pbuf_21_1); EXEC_UNMAP_ROM(pbuf_21_2);

EXEC_FREE_MEM(pbuff20);
WAIT_4_COMPLETION();
memcpy(pOut0, GET_BUFF_ADDR(out0), NUM_BYTES(out0));
EXEC_FREE_MEM(out0);
}
