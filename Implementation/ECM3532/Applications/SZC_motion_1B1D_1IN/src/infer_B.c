#include "executor_public.h"

DECL_CONST_INT_ROM(pbuf_1_1_B,108);
DECL_CONST_INT_ROM(pbuf_1_2_B,4);
DECL_CONST_INT_ROM(pbuf_5_1_B,96);
DECL_CONST_INT_ROM(pbuf_5_2_B,8);
DECL_CONST_INT_ROM(pbuf_9_1_B,384);
DECL_CONST_INT_ROM(pbuf_9_2_B,16);
DECL_CONST_INT_ROM(pbuf_13_1_B,1536);
DECL_CONST_INT_ROM(pbuf_13_2_B,32);
DECL_CONST_INT_ROM(pbuf_17_1_B,768);
DECL_CONST_INT_ROM(pbuf_17_2_B,8);
DECL_CONST_INT_ROM(pbuf_21_1_B,192);
DECL_CONST_INT_ROM(pbuf_21_2_B,6);

DECL_BUF_M3_PERSISTENT(pbuff0_B,1152);
DECL_BUF_M3_PERSISTENT(pbuff1_B,512);
DECL_BUF_M3_PERSISTENT(pbuff2_B,512);
DECL_BUF_M3_PERSISTENT(pbuff3_B,256);
DECL_BUF_M3_PERSISTENT(pbuff4_B,256);
DECL_BUF_M3_PERSISTENT(pbuff5_B,512);
DECL_BUF_M3_PERSISTENT(pbuff6_B,512);
DECL_BUF_M3_PERSISTENT(pbuff7_B,256);
DECL_BUF_M3_PERSISTENT(pbuff8_B,256);
DECL_BUF_M3_PERSISTENT(pbuff9_B,512);
DECL_BUF_M3_PERSISTENT(pbuff10_B,512);
DECL_BUF_M3_PERSISTENT(pbuff11_B,256);
DECL_BUF_M3_PERSISTENT(pbuff12_B,256);
DECL_BUF_M3_PERSISTENT(pbuff13_B,512);
DECL_BUF_M3_PERSISTENT(pbuff14_B,512);
DECL_BUF_M3_PERSISTENT(pbuff15_B,256);
DECL_BUF_M3_PERSISTENT(pbuff16_B,256);
DECL_BUF_M3_PERSISTENT(pbuff17_B,64);
DECL_BUF_M3_PERSISTENT(pbuff18_B,64);
DECL_BUF_M3_PERSISTENT(pbuff19_B,32);
DECL_BUF_M3_PERSISTENT(pbuff20_B,32);
DECL_BUF_M3_PERSISTENT(out0_B,6);
DECL_BUF_M3_PERSISTENT(inp0_B,1152);

DECL_BUF_M3_SCRATCH(pbuf_1_4_B,54);
DECL_BUF_M3_SCRATCH(pbuf_5_4_B,24);
DECL_BUF_M3_SCRATCH(pbuf_9_4_B,48);
DECL_BUF_M3_SCRATCH(pbuf_13_4_B,96);
DECL_BUF_M3_SCRATCH(pbuf_17_4_B,192);


void infer_B(q7_t *pIn0, q7_t *pOut0) { 

EXEC_ALLOC_MEM(inp0_B);
memcpy(GET_BUFF_ADDR(inp0_B), pIn0, NUM_BYTES(inp0_B));

EXEC_ALLOC_MEM(pbuff0_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff0_B), GET_BUFF_ADDR(inp0_B), NUM_BYTES(inp0_B));
EXEC_FREE_MEM(inp0_B);
EXEC_ALLOC_MEM(pbuff1_B);
conv2d_opt opt1 = { 
.in_rows = 1, .in_cols = 128, .in_depth = 9, .num_filt = 4, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 8,
.out_rows = 1, .out_cols = 128,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_1_4_B);
EXEC_MAP_ROM(pbuf_1_1_B); EXEC_MAP_ROM(pbuf_1_2_B);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff0_B, &pbuf_1_1_B, &pbuf_1_2_B, &pbuff1_B, &pbuf_1_4_B, &opt1);
EXEC_UNMAP_ROM(pbuf_1_1_B); EXEC_UNMAP_ROM(pbuf_1_2_B);

EXEC_FREE_MEM(pbuff0_B);
EXEC_ALLOC_MEM(pbuff2_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff2_B), GET_BUFF_ADDR(pbuff1_B), NUM_BYTES(pbuff1_B));
EXEC_FREE_MEM(pbuff1_B);
EXEC_ALLOC_MEM(pbuff3_B);
pool2d_opt opt3 = { .in_rows = 128, .in_cols = 1, .in_depth = 4, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 64, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff2_B, &pbuff3_B, &opt3);

EXEC_FREE_MEM(pbuff2_B);
EXEC_ALLOC_MEM(pbuff4_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff4_B), GET_BUFF_ADDR(pbuff3_B), NUM_BYTES(pbuff3_B));
EXEC_FREE_MEM(pbuff3_B);
EXEC_ALLOC_MEM(pbuff5_B);
conv2d_opt opt5 = { 
.in_rows = 1, .in_cols = 64, .in_depth = 4, .num_filt = 8, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 64,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_5_4_B);
EXEC_MAP_ROM(pbuf_5_1_B); EXEC_MAP_ROM(pbuf_5_2_B);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff4_B, &pbuf_5_1_B, &pbuf_5_2_B, &pbuff5_B, &pbuf_5_4_B, &opt5);
EXEC_UNMAP_ROM(pbuf_5_1_B); EXEC_UNMAP_ROM(pbuf_5_2_B);

EXEC_FREE_MEM(pbuff4_B);
EXEC_ALLOC_MEM(pbuff6_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff6_B), GET_BUFF_ADDR(pbuff5_B), NUM_BYTES(pbuff5_B));
EXEC_FREE_MEM(pbuff5_B);
EXEC_ALLOC_MEM(pbuff7_B);
pool2d_opt opt7 = { .in_rows = 64, .in_cols = 1, .in_depth = 8, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 32, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff6_B, &pbuff7_B, &opt7);

EXEC_FREE_MEM(pbuff6_B);
EXEC_ALLOC_MEM(pbuff8_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff8_B), GET_BUFF_ADDR(pbuff7_B), NUM_BYTES(pbuff7_B));
EXEC_FREE_MEM(pbuff7_B);
EXEC_ALLOC_MEM(pbuff9_B);
conv2d_opt opt9 = { 
.in_rows = 1, .in_cols = 32, .in_depth = 8, .num_filt = 16, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 32,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_9_4_B);
EXEC_MAP_ROM(pbuf_9_1_B); EXEC_MAP_ROM(pbuf_9_2_B);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff8_B, &pbuf_9_1_B, &pbuf_9_2_B, &pbuff9_B, &pbuf_9_4_B, &opt9);
EXEC_UNMAP_ROM(pbuf_9_1_B); EXEC_UNMAP_ROM(pbuf_9_2_B);

EXEC_FREE_MEM(pbuff8_B);
EXEC_ALLOC_MEM(pbuff10_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff10_B), GET_BUFF_ADDR(pbuff9_B), NUM_BYTES(pbuff9_B));
EXEC_FREE_MEM(pbuff9_B);
EXEC_ALLOC_MEM(pbuff11_B);
pool2d_opt opt11 = { .in_rows = 32, .in_cols = 1, .in_depth = 16, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 16, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff10_B, &pbuff11_B, &opt11);

EXEC_FREE_MEM(pbuff10_B);
EXEC_ALLOC_MEM(pbuff12_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff12_B), GET_BUFF_ADDR(pbuff11_B), NUM_BYTES(pbuff11_B));
EXEC_FREE_MEM(pbuff11_B);
EXEC_ALLOC_MEM(pbuff13_B);
conv2d_opt opt13 = { 
.in_rows = 1, .in_cols = 16, .in_depth = 16, .num_filt = 32, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 16,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_13_4_B);
EXEC_MAP_ROM(pbuf_13_1_B); EXEC_MAP_ROM(pbuf_13_2_B);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff12_B, &pbuf_13_1_B, &pbuf_13_2_B, &pbuff13_B, &pbuf_13_4_B, &opt13);
EXEC_UNMAP_ROM(pbuf_13_1_B); EXEC_UNMAP_ROM(pbuf_13_2_B);

EXEC_FREE_MEM(pbuff12_B);
EXEC_ALLOC_MEM(pbuff14_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff14_B), GET_BUFF_ADDR(pbuff13_B), NUM_BYTES(pbuff13_B));
EXEC_FREE_MEM(pbuff13_B);
EXEC_ALLOC_MEM(pbuff15_B);
pool2d_opt opt15 = { .in_rows = 16, .in_cols = 1, .in_depth = 32, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 8, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff14_B, &pbuff15_B, &opt15);

EXEC_FREE_MEM(pbuff14_B);
EXEC_ALLOC_MEM(pbuff16_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff16_B), GET_BUFF_ADDR(pbuff15_B), NUM_BYTES(pbuff15_B));
EXEC_FREE_MEM(pbuff15_B);
EXEC_ALLOC_MEM(pbuff17_B);
conv2d_opt opt17 = { 
.in_rows = 1, .in_cols = 8, .in_depth = 32, .num_filt = 8, 
.filt_rows = 1, .filt_cols = 3, .row_pad = 0, .col_pad = 1, 
.row_stride = 1, .col_stride = 1, .bias_lshift = 0, .out_rshift = 7,
.out_rows = 1, .out_cols = 8,.act_min = 0, .act_max = 127 };

EXEC_ALLOC_MEM(pbuf_17_4_B);
EXEC_MAP_ROM(pbuf_17_1_B); EXEC_MAP_ROM(pbuf_17_2_B);
Exec_conv2d_q7(EXEC_HW_ID_M3, &pbuff16_B, &pbuf_17_1_B, &pbuf_17_2_B, &pbuff17_B, &pbuf_17_4_B, &opt17);
EXEC_UNMAP_ROM(pbuf_17_1_B); EXEC_UNMAP_ROM(pbuf_17_2_B);

EXEC_FREE_MEM(pbuff16_B);
EXEC_ALLOC_MEM(pbuff18_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff18_B), GET_BUFF_ADDR(pbuff17_B), NUM_BYTES(pbuff17_B));
EXEC_FREE_MEM(pbuff17_B);
EXEC_ALLOC_MEM(pbuff19_B);
pool2d_opt opt19 = { .in_rows = 8, .in_cols = 1, .in_depth = 8, .filt_rows = 2, .filt_cols = 1, .row_pad = 0, 
.col_pad = 0, .row_stride = 2, .col_stride = 1, 
.out_lshift = 0, .out_rows = 4, .out_cols = 1,.act_max = 127, .act_min = 0};

Exec_maxpool_q7(EXEC_HW_ID_M3, &pbuff18_B, &pbuff19_B, &opt19);

EXEC_FREE_MEM(pbuff18_B);
EXEC_ALLOC_MEM(pbuff20_B);
WAIT_4_COMPLETION();
memcpy(GET_BUFF_ADDR(pbuff20_B), GET_BUFF_ADDR(pbuff19_B), NUM_BYTES(pbuff19_B));
EXEC_FREE_MEM(pbuff19_B);
EXEC_ALLOC_MEM(out0_B);
fc_opt opt21 = { 
.filt_rows = 6, .filt_cols = 32, .bias_shift = 0, .out_shift = 6, .input_length = 32,
.act_min = 0, .act_max= 127};

EXEC_MAP_ROM(pbuf_21_1_B); EXEC_MAP_ROM(pbuf_21_2_B);
Exec_fully_connected_q7(EXEC_HW_ID_M3, &pbuff20_B, &pbuf_21_1_B, &pbuf_21_2_B, &out0_B, &opt21);
EXEC_UNMAP_ROM(pbuf_21_1_B); EXEC_UNMAP_ROM(pbuf_21_2_B);

EXEC_FREE_MEM(pbuff20_B);
WAIT_4_COMPLETION();
memcpy(pOut0, GET_BUFF_ADDR(out0_B), NUM_BYTES(out0_B));
EXEC_FREE_MEM(out0_B);
}
