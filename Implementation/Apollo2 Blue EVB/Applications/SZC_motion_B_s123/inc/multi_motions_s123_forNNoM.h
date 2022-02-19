#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0 {-9, -19, -1, -1, -48, -30, -59, 3, 0, -20, 39, -48, 8, 3, -49, -13, 20, -19, -27, -20, -50, -66, 14, -3, -76, -9, 5, -5, 21, -42, 27, -1, -24, 69, -30, 31, -51, -13, -22, -9, 4, -37, 12, -52, 34, -45, 18, -12, 5, 21, -73, 38, -76, -56, 19, -70, 6, 26, 43, 27, 29, -2, 20, -47, -11, -9, -8, 13, 22, 40, 64, -12, -41, -16, -46, -34, -27, -35, -1, 43, 17, -5, 12, -48, -15, 27, -35, 40, 19, -57, 37, -14, 48, -19, -43, 24, 4, 55, 37, 70, -18, -6, -33, -10, -70, -7, 20, 45}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS {7}

#define TENSOR_CONV1D_BIAS_0 {1, 24, -12, -23}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS {7}

#define CONV1D_BIAS_LSHIFT {0}

#define CONV1D_OUTPUT_RSHIFT {9}

#define TENSOR_CONV1D_1_KERNEL_0 {50, 13, -2, -49, 2, 28, -98, -82, 32, 8, -51, 16, -56, -19, -10, -12, 25, 44, 11, -33, -20, -4, 30, -86, -50, -78, 19, 11, -22, -71, -2, -21, 22, -53, 28, 38, 6, -9, 21, -31, -35, 39, 7, 12, -3, 26, -9, 36, -41, -24, -79, -55, -46, -23, -22, 37, 0, -3, 0, 23, 44, 31, -77, -21, -10, -20, 32, 50, -63, -30, 29, -40, -8, -35, -79, -83, -2, 36, 0, -1, -23, -17, 16, 43, 0, -79, 2, -45, 39, -66, -25, 2, 65, -55, 25, -3}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS {7}

#define TENSOR_CONV1D_1_BIAS_0 {13, 7, 12, -60, -16, -54, -18, -39}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS {7}

#define CONV1D_1_BIAS_LSHIFT {0}

#define CONV1D_1_OUTPUT_RSHIFT {7}

#define TENSOR_CONV1D_2_KERNEL_0 {30, 20, -60, -7, -20, -27, 9, -67, 31, -18, -11, -48, -11, -4, -61, -1, -1, -51, -33, -30, -25, 1, 6, -11, 27, -18, -15, 0, -29, -1, 14, -2, -55, -45, -24, 51, 27, -12, -36, 12, -63, -24, 24, -15, -11, -60, -38, 37, -38, -10, -13, -42, -13, -42, -2, -3, 8, -39, -6, -20, 23, 7, 0, -31, 10, -27, -17, -19, -35, -5, -6, -6, -32, 3, -19, -17, -21, 2, -25, 2, 2, 10, 1, -41, 26, -16, -39, -23, 15, -45, -36, 5, 27, -4, 4, -21, -20, 24, -13, -14, -26, -29, -37, -14, 10, -35, 5, -11, -31, -35, -19, -17, -6, -9, -38, -33, -30, 17, 23, -12, -37, 14, -62, -52, -28, -10, -9, -84, -13, -22, -98, 6, 7, -1, -69, -65, 13, -74, 48, -45, 24, 34, 46, 12, 21, 16, 15, -36, 16, 7, 9, -23, -20, 44, -28, 12, 2, -24, -93, 26, -8, -33, 13, 8, 7, 37, 25, 12, -27, 4, -16, -10, 70, 44, 55, 6, -49, 38, -2, 4, 6, -29, 47, -16, -44, 43, -3, 14, -23, 30, -8, -31, 17, 16, 12, -22, 17, -11, -47, -4, 15, -8, -22, -23, -59, -30, 43, 28, 11, -32, 15, 27, -59, 35, -12, 47, 47, 12, -43, 32, 8, 40, 42, -62, -55, -35, -31, 6, -29, 42, 4, -24, -57, -11, 22, -72, -60, 9, -64, -8, 5, -36, 25, 34, -45, -5, -67, 3, -33, -42, 15, -23, 6, 16, 54, 22, 25, -8, -21, -47, 8, 25, 9, -6, -36, 24, 13, 30, 15, 36, -14, -40, 27, -13, -9, 8, -7, 35, -33, 25, -61, -25, 12, -83, -16, 26, 8, -17, -6, 11, -38, 23, 30, -16, -14, 10, 25, -10, 5, 11, -57, 43, -37, -32, -7, 23, -16, 8, 5, 8, -17, -49, -13, 0, 45, -22, -13, -34, -60, -12, 16, 4, 58, -43, -1, -5, 22, -14, 20, 11, -31, -24, -6, 2, 11, 0, -38, -29, -37, -17, -34, 7, 2, 9, -35, -17, -5, -23, -40, -8, 17, -35, -48, -24, 13, -8, 13, 7, -50, -14, -12, -6, -24, -42, -38, 69, -8, 17, -6, -12, 54, 14, -35, -31, -37, -28, 0, -25, -40, 43, 14, -74, 16, -18}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS {7}

#define TENSOR_CONV1D_2_BIAS_0 {56, -53, -31, -21, -28, 49, 3, -50, -25, -53, -13, 22, 21, -5, -15, -33}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS {7}

#define CONV1D_2_BIAS_LSHIFT {0}

#define CONV1D_2_OUTPUT_RSHIFT {6}

#define TENSOR_CONV1D_3_KERNEL_0 {-27, -28, 1, -5, 6, -57, -40, 27, 0, 30, -10, -50, 3, -33, -17, -32, -3, 1, 31, -10, -23, 1, -33, 40, 20, 1, 13, -2, -8, -17, 9, -6, -37, -16, 3, -5, 11, -9, -23, -64, 1, -2, -16, -15, 16, 7, 17, 6, -10, -2, -21, 17, 24, 17, 12, -23, 6, -5, 21, -6, 18, 9, 17, 51, -4, 9, 19, -25, 1, 29, 3, 32, 15, -46, -22, -25, -4, -9, 6, 27, 15, 19, -30, -5, -18, 114, -19, -6, -20, -34, 22, -8, 10, -8, 16, 10, 17, 2, -39, 25, -2, 14, 2, 4, -14, -44, -24, 19, 40, -19, 3, 22, 47, -32, 0, -2, 1, 31, 4, -57, -3, -59, -2, -4, 25, -7, -28, -9, 46, -30, -1, -28, 28, 21, 28, -22, 20, -45, 30, 18, 1, 7, 49, -47, -17, -16, -44, -2, -5, 39, -26, -9, 6, 3, 9, 4, -25, -14, 11, 5, 6, -57, -18, -18, 13, -13, 11, 26, -24, -62, -16, -24, -13, -2, 3, -55, 7, -81, 6, -22, 16, -32, 20, 37, -16, -56, -10, -18, 23, -31, -31, -34, 22, -17, 10, -20, 14, -21, -23, -10, 25, 54, -20, -14, -9, 4, -6, -12, -26, 18, -14, 1, -9, -18, -3, -49, -16, -75, 4, -51, -14, 36, 7, -15, 5, 13, -33, -1, -11, 29, -19, -13, 12, 24, 13, 23, 25, 6, 8, -57, 1, -32, 9, 4, 50, -8, -49, 0, -37, -34, -3, -12, -68, -46, -16, -33, 73, 20, 54, 0, -45, 3, -53, -100, 18, -24, -17, -41, -37, -54, -14, -29, -22, -45, 22, -12, -18, -59, 18, -51, 1, -29, -25, -38, -17, -17, 27, -59, 19, -12, -13, 1, -10, 10, 13, -36, -21, 11, -35, -11, 10, 19, -16, -44, -9, 1, 19, 12, 15, -37, -21, -28, 20, -20, -25, -26, 5, -10, 12, -50, -13, 12, 5, -2, 23, 8, 7, 20, -17, 13, -7, -10, -25, -14, 4, -58, 39, 4, -45, -40, 12, -3, 14, 1, -19, 46, 29, 44, -10, -22, 12, 39, -32, -24, 46, 24, -15, 43, 15, -31, 1, 18, -16, -24, -39, -56, -3, 21, -10, -14, 35, -19, 0, 1, -5, 26, 0, 6, 40, 19, 17, -4, -24, 6, 32, 6, 10, -60, -9, -33, 16, -12, 50, -44, -2, -37, -19, 23, 8, -48, -30, 2, 2, -31, -19, 73, 7, 16, -6, -17, 12, 5, -21, 39, 3, 67, 44, -9, -38, 4, 1, 3, -26, 3, -35, 2, -42, -30, -1, -70, 17, -30, 28, 1, 6, 22, -22, -9, 5, 3, -16, -11, -7, -1, -14, 12, 19, 2, 3, -9, -23, 15, 7, 44, 21, -32, 18, -18, 10, -29, -18, 5, -7, -33, 18, 27, -58, -14, 10, 1, -19, -44, 12, 12, 14, -7, -19, -6, -25, -28, -8, 15, -30, 14, -14, -57, -19, -44, -2, -6, -16, -30, 8, -41, -28, 41, -24, 24, -14, -5, 11, 13, -34, -19, -41, 14, -28, -8, -34, -39, 11, -3, -2, -33, 31, 8, 15, -24, -3, 7, 1, 27, 27, 3, 1, -23, 37, 19, -29, -20, -26, -14, 26, -18, -28, -36, -4, -8, 1, -20, -27, -10, -4, -12, -28, -15, -17, 11, -13, 3, -12, -22, -34, -1, 9, -10, -5, -27, -25, -17, -22, -12, 13, -10, -22, -45, 11, -24, -17, 11, -3, -25, 6, 5, -5, 10, 2, 1, 32, -14, -12, -35, 11, -46, 28, -24, -18, -12, -3, -9, 24, 56, -22, 24, 12, 9, 2, 38, -1, -15, -13, 61, 14, 15, -14, -13, -5, 10, -26, 8, -14, 7, -20, -23, -8, 10, 25, 0, -7, 28, -26, -18, 15, -45, -15, -7, 15, 2, 8, 5, 8, 3, -13, -64, 7, 12, -10, -5, -32, -75, -22, -6, 6, 22, -15, 15, 11, -35, 1, -30, 11, -25, 10, 23, 17, 5, 4, 24, -17, 24, 31, 16, 16, 19, 5, 54, 1, -7, -12, -8, 16, 39, 10, -44, 11, 13, -1, 1, 0, -39, -31, -16, -35, -4, -17, -12, 12, 20, 18, 29, 7, -2, 0, -9, -12, 7, -25, -38, -26, 10, -5, -11, 8, 36, 10, -14, -28, 5, 4, 14, 16, 4, -28, -41, 13, -19, -3, 20, -39, -38, -32, 25, 16, 7, 22, 25, -1, -30, 10, -71, -3, -17, -11, -7, -16, -69, -14, 6, 30, 5, -29, 84, 3, -8, 9, -57, 21, 0, -5, 22, -18, -54, -13, 22, 13, -26, -16, -23, -31, 24, 14, -51, 17, -6, -52, 10, -17, 10, -29, -48, -32, -1, -34, 33, -11, -16, -2, -1, -11, 0, -85, -36, -11, 21, 6, -33, -29, 27, -23, -14, -88, -51, 9, -16, -28, -49, 5, -16, -33, -4, -67, -34, -64, -10, -17, -26, -53, -39, -59, -9, -1, -4, -37, -29, -23, -45, 5, -52, 5, -22, 9, -14, -27, 3, -38, -70, -34, 6, 11, 9, -2, -45, 28, -34, -22, 9, -21, 4, 8, -28, -35, 47, -28, 21, -1, -11, 6, 9, 8, -17, -29, 13, 6, -27, 5, -27, 14, 17, -46, -22, -25, -24, 11, 35, -37, -5, -26, 14, -20, -16, 13, -42, -29, 15, 44, 36, -30, -25, 24, 17, -24, -62, -11, -28, 3, -6, 5, 27, 0, -26, 14, -7, -10, -38, 22, 17, 3, -74, -25, -18, 30, 8, 19, 66, -23, -22, 4, -50, 7, -10, 4, -40, 8, -13, 1, -16, -4, -39, -2, 16, 15, 12, -14, -2, -19, 19, -9, -19, -25, -34, -2, -15, 17, -17, -13, 0, 9, -47, -21, -14, -24, 4, -29, 14, 15, -1, 8, 1, 10, 22, 20, -37, 13, 28, 21, 8, 23, 10, 7, 27, -24, -12, 12, -7, 2, -28, -31, 1, -30, -42, -4, -52, -2, -13, 30, -72, 23, -1, 13, -12, -26, -19, 8, -9, -12, 41, -32, 8, 17, -21, 29, -19, 10, -24, -39, 3, -6, 20, 19, -2, -37, 38, 5, -11, 1, -4, -9, 16, -40, 21, 13, 3, -1, 44, 8, 11, -38, 29, 2, 9, 14, 4, -31, -4, -33, -49, 6, -43, -10, -5, -69, -33, -51, -8, -17, -35, -36, 12, -53, -30, 23, -73, 6, 2, -26, -37, -73, -69, -12, -28, -64, -18, 4, -33, -51, -27, 33, 17, 11, 8, -18, 3, -20, 75, 29, 3, -26, -17, 9, -20, 19, 40, 4, -15, -13, 5, 18, -14, 8, 3, 7, -61, 15, -3, -6, 16, -14, -24, 29, -4, -10, 23, -10, 31, 6, -14, -3, 26, 9, -16, 12, -9, 13, -24, 5, 5, -10, -11, 28, -23, 16, -52, 13, -21, -32, -10, -22, 23, 6, -24, 19, -60, -18, -17, 34, 10, 10, 19, -29, -22, -13, -35, -8, -11, -16, -40, 3, -57, 6, 20, -34, -13, -6, -20, -16, -60, -9, -8, -5, 9, 5, -17, 7, 13, -34, 10, 5, -12, -23, 10, -22, -44, 8, -37, 29, -38, 29, 22, 32, -51, -20, -1, 19, -11, -19, -47, -28, -52, -27, -10, 20, -15, 7, 1, 33, 16, -6, -6, -26, -15, 5, -17, 37, -53, 18, -8, 12, -21, 25, -16, -35, 18, 8, 7, 6, 14, 3, -36, 17, -38, 24, -6, 11, -16, 11, -68, -1, 24, 27, 11, 20, 16, -41, -19, 20, -41, -19, -14, -36, 21, 3, -20, 5, 4, -6, -18, 14, 13, -26, -14, 14, -44, 8, -37, -40, -28, 8, -42, -24, 3, 40, 14, 3, -24, 7, -24, 6, -33, -2, 4, 5, -24, 5, 14, -23, 39, -11, 20, -9, -20, 20, 7, -17, 72, -24, 18, 5, -3, 3, 0, -30, -32, 16, -3, -11, -62, -13, -28, 30, -1, 0, -18, -1, 22, 16, 35, -27, -22, -37, -26, 19, -3, -23, -49, -24, -23, -50, -9, -12, -1, -17, -2, -19, -3, -10, -2, -31, 6, -46, -7, -41, 17, -16, -3, -48, 4, -24, -61, 10, -24, 8, -33, 15, -4, 10, -9, -16, 14, -8, -41, -11, -9, -24, -45, -20, 14, 22, -12, -2, 14, 21, 42, -6, 34, -24, 34, 26, 10, -1, -15, -22, -30, 5, 2, 37, 36, -3, -29, 3, 27, -26, -25, 4, -8, -24, 10, -26, 18, 30, -25, -7, 19, 9, -1, 0, 12, -9, -15, -28, -38, 6, 35, -2, 1, 1, 3, 10, -9, -31, 42, 16, 17, 6, -4, -17, -2, -8, -13, -34, 13, 22, 4, -14, 41, 32, -53, 22, -47, -9, -32, -16, 31, 19, -17, -35, 43, -36, 0, -9, -10, 34, -22, -2, -6, -20, -25, -5, 5, 6, 55, 38, -28, -4, 1, 33, 37, -4, 25, -21, 26, 24, -2, -19, 14, 5, 26, 9, -30, -11, -6, -16, 41, -4, 71, 1, -53, -16, -19, 13, 5, -1, -12, 31, 22, -5, 3, -11, 9, 22, 24, -2, -34, -11, 11, 13, -9, -17, -41, 23, -10, -2, 2, 11, 4, -21, -39, -33, -11, -8, -7, -18, -10, 38, -42, 10, 11, 2, 0, -13, 110, 26, -29, -2, -69, 17, 45, 19, 44, 2, -2, 11, -1, 14, -21, 7, -26, -29, -46, -14, -9, -20, -27, -4, -27, -59, -36}

#define TENSOR_CONV1D_3_KERNEL_0_DEC_BITS {7}

#define TENSOR_CONV1D_3_BIAS_0 {-47, 12, 87, -3, -38, -43, -6, -84, 31, -16, -34, -16, -52, -30, -11, -3, -31, -30, -18, -70, -17, -46, -52, -9, 54, -23, -53, -2, -52, 18, -18, 116}

#define TENSOR_CONV1D_3_BIAS_0_DEC_BITS {7}

#define CONV1D_3_BIAS_LSHIFT {0}

#define CONV1D_3_OUTPUT_RSHIFT {6}

#define TENSOR_CONV1D_4_KERNEL_0 {-6, -6, 16, -31, 5, 24, 9, 10, 35, -27, 29, 11, 42, -12, 5, -18, -16, -17, -63, -21, -43, 45, -27, -13, -31, 16, 18, -10, -9, -6, -33, 19, 45, 7, -21, -64, 14, 8, -14, -5, 9, 19, -8, 34, 14, 16, -12, 25, 44, -73, -46, 28, -11, 55, 22, -16, -17, 16, -6, 27, 26, 62, -23, 32, -3, -6, -6, -19, 15, 48, 14, -3, 5, 23, 31, 7, 35, 22, 8, 16, -1, -58, -19, 9, -28, 19, -5, 4, 23, 29, 21, 35, -8, 0, -25, -30, -6, 19, 2, 24, -1, 5, -19, 41, 44, -12, 34, 17, -18, -16, -6, -40, -17, 33, 69, -16, -23, -9, 25, -18, -3, -6, 2, -22, 26, -15, 47, -84, 26, 4, 9, 8, 34, -29, -8, 17, -16, 13, 27, -13, 20, 0, 5, -33, 12, 10, 12, 17, -1, -27, 24, -2, 13, -31, 13, 2, -9, -22, 32, -11, -35, 20, 29, 6, -17, 8, 22, 42, 15, 19, -8, 8, -10, 7, 18, 26, 18, 21, 17, -8, 19, 11, 25, -1, 19, 7, 20, 55, 18, 16, 1, -31, 7, -15, -2, -32, 6, -67, -30, -47, -11, 14, 0, -2, -28, -20, 1, 20, -12, -11, 2, -25, -15, -53, 6, 2, -28, 24, -25, -46, -15, 27, -10, -51, 25, -18, -34, 0, -29, -72, -17, -8, 16, 20, 25, 32, 20, 20, -12, -8, 11, 20, -22, 6, 14, 17, 15, -10, 2, -9, -21, -23, 16, -4, -8, -30, -16, -29, -14, 6, 1, 9, 22, -15, -9, -6, -7, -13, -27, 13, 12, -47, 44, 17, 4, 20, 9, -18, -31, -8, 21, 13, 12, -30, -23, 4, -9, -9, -18, -30, 29, -23, 0, 32, 40, -11, 13, 27, -13, -5, -47, 23, 1, 34, 63, -8, 28, -11, 44, -22, 11, 3, 24, -64, -10, 15, 13, -21, 7, 35, 18, 27, 20, -44, -65, -40, 1, -15, 50, -21, 0, 9, 6, 11, -15, -35, -32, -27, -5, 4, -27, 2, -14, -34, 20, -6, -6, 13, -24, 22, 11, 9, 29, 30, 53, -70, -4, -1, 16, -49, 55, 2, 9, -21, -13, -39, 43, -1, -16, -18, -28, -31, -29, 17, 23, -41, 42, -48, -11, 14, -32, 21, 0, 23, -48, -18, 2, -13, -5, -51, 18, -95, -6, 20, -20, 15, 15, -22, -22, -1, -12, -31, -25, 15, 11, -56, -13, -10, 2, 14, -6, 45, -57, -14, -24, 19, -38, -7, -12, -33, -12, 47, 6, -107, -32, -13, 13, 32, 16, -31, -12, 37, 10, 5, -47, -20, -22, 19, -32, -29, 11, -12, -20, 26, -47, 13, 0, 64, -2, 16, 32, -41, 17, 62, 1, -7, 30, -29, 18, -2, -1, -5, -5, 34, -12, 35, -11, 24, 15, 1, -23, 7, -16, 31, 29, 16, 1, 9, -22, 116, -28, -27, -32, 8, -23, 25, 13, -28, 23, -7, -35, -30, -19, -4, -34, -2, -22, 3, -15, 3, -29, -8, -5, 16, 4, -16, -2, -5, -14, 3, -41, 14, 17, 11, -40, 16, 16, 80, -18, -36, 3, 10, 5, -9, 0, -39, -2, -7, -21, -27, 25, 9, -20, 18, 7, -27, -17, -31, 0, 24, -17, 12, -7, -38, -12, 11, 20, 2, -18, 13, -4, 14, 14, -29, -7, -27, 8, -41, 9, 10, 15, -23, 20, -21, 4, 27, -40, -3, -33, 9, -22, 4, -43, 6, -25, -37, -31, 8, -31, 19, -14, -27, 5, -12, 1, -31, 0, 1, 1, -31, 3, -18, 19, -35, 4, -12, -7, -25, -5, -10, 0, 16, -18, -41, 13, -29, -35, 18, -19, -26, -30, 11, 20, -13, 3, -47, -32, -25, 9, 2, 6, 10, 15, 11, -25, 7, 16, -33, 21, -13, 21, -12, -28, -20, -22, -21, 18, -20, -33, -3, -24, 4, 22, 16, -14, -18, -10, -27, 15, -1, -29, -35, 9, -16, 21, -31, -20, 13, -26, -26, -20, -70, -26, 19, -21, 7, 13, -21, 11, -26, -2, 6, -9, -26, -21, 26, 8, -80, -24, -28, 24, 19, 23, -8, -13, -29, 27, -6, 12, -15, -2, -12, 12, -17, -25, -11, -8, -5, -58, 33, -28, -41, 13, -11, 10, 2, -15, -7, -14, 38, -1, -4, 60, -18, -11, 9, 22, 28, 11, 26, 17, -30, 13, 26, 21, 13, -6, -6, -26, 29, -11, 43, -19, -20, -12, 20, -21, 24, 19, -18, -15, 28, -19, -12, 30, 9, -7, 10, 18, 11, -5, 18, -42, -23, -24, 9, -14, 14, -3, 1, -6, 29, -7, -3, 12, 30, 10, 22}

#define TENSOR_CONV1D_4_KERNEL_0_DEC_BITS {7}

#define TENSOR_CONV1D_4_BIAS_0 {-21, -43, -13, 91, 83, -15, -7, -24}

#define TENSOR_CONV1D_4_BIAS_0_DEC_BITS {7}

#define CONV1D_4_BIAS_LSHIFT {0}

#define CONV1D_4_OUTPUT_RSHIFT {7}

#define TENSOR_DENSE_KERNEL_0 {-31, -22, 5, -9, -7, 19, 16, -13, 6, -2, -19, 10, -5, 2, -28, 32, -16, 6, 22, 13, -32, 8, -13, -14, 17, 18, 5, 3, -19, 19, 23, -20, -16, -1, 7, 20, 12, -22, 0, -15, 17, 11, -33, 21, 12, -57, -22, -10, 26, 0, -2, 3, -6, 9, -8, -17, -12, 17, -21, 9, -17, 2, 14, 5, -12, 12, 15, -3, 25, -17, -18, 14, 12, 8, -43, 13, 3, -21, -38, -14, -16, -21, 22, -7, -14, 19, 5, 12, 19, -11, 13, 4, 20, -13, -16, -21, -22, -18, 17, -12, 10, 14, 19, -4, 25, 7, -30, 1, 6, -13, -32, 28, -32, -46, -17, -20, -35, 36, 1, 22, -4, -2, 10, 10, -20, -8, 11, 32, -18, -17, 8, 19, 29, -10, 2, 13, -10, -15, 20, 23, 26, -26, 8, -18, 8, -20, 1, 29, 19, -1, -1, 2, -26, -33, 14, 4, 21, 4, 20, -7, 8, -13, -16, -25, 1, -3, -24, 21, -11, -22, 11, -73, 22, -12, -9, 18, 18, -34, -17, -61, 18, 28, -6, 11, 10, -37, 17, -58, 27, 10, -25, -22}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS {6}

#define TENSOR_DENSE_BIAS_0 {-16, -11, -25, 24, 45, -31}

#define TENSOR_DENSE_BIAS_0_DEC_BITS {6}

#define DENSE_BIAS_LSHIFT {0}

#define DENSE_OUTPUT_RSHIFT {6}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC 0
#define INPUT_1_OUTPUT_OFFSET 0
#define CONV1D_OUTPUT_DEC -2
#define CONV1D_OUTPUT_OFFSET 0
#define RE_LU_OUTPUT_DEC -2
#define RE_LU_OUTPUT_OFFSET 0
#define MAX_POOLING1D_OUTPUT_DEC -2
#define MAX_POOLING1D_OUTPUT_OFFSET 0
#define CONV1D_1_OUTPUT_DEC -2
#define CONV1D_1_OUTPUT_OFFSET 0
#define RE_LU_1_OUTPUT_DEC -2
#define RE_LU_1_OUTPUT_OFFSET 0
#define MAX_POOLING1D_1_OUTPUT_DEC -2
#define MAX_POOLING1D_1_OUTPUT_OFFSET 0
#define CONV1D_2_OUTPUT_DEC -1
#define CONV1D_2_OUTPUT_OFFSET 0
#define RE_LU_2_OUTPUT_DEC -1
#define RE_LU_2_OUTPUT_OFFSET 0
#define MAX_POOLING1D_2_OUTPUT_DEC -1
#define MAX_POOLING1D_2_OUTPUT_OFFSET 0
#define CONV1D_3_OUTPUT_DEC 0
#define CONV1D_3_OUTPUT_OFFSET 0
#define RE_LU_3_OUTPUT_DEC 0
#define RE_LU_3_OUTPUT_OFFSET 0
#define MAX_POOLING1D_3_OUTPUT_DEC 0
#define MAX_POOLING1D_3_OUTPUT_OFFSET 0
#define CONV1D_4_OUTPUT_DEC 0
#define CONV1D_4_OUTPUT_OFFSET 0
#define RE_LU_4_OUTPUT_DEC 0
#define RE_LU_4_OUTPUT_OFFSET 0
#define MAX_POOLING1D_4_OUTPUT_DEC 0
#define MAX_POOLING1D_4_OUTPUT_OFFSET 0
#define FLATTEN_OUTPUT_DEC 0
#define FLATTEN_OUTPUT_OFFSET 0
#define DENSE_OUTPUT_DEC 0
#define DENSE_OUTPUT_OFFSET 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data[1152] = {0};

const nnom_shape_data_t tensor_input_1_0_dim[] = {128, 9};
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

const nnom_shape_data_t tensor_conv1d_kernel_0_dim[] = {3, 9, 4};
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

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim[] = {3, 4, 8};
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

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim[] = {8};
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
    .filter_size = 8,
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

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim[] = {3, 8, 16};
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

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim[] = {16};
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
    .filter_size = 16,
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
const int8_t tensor_conv1d_3_kernel_0_data[] = TENSOR_CONV1D_3_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_3_kernel_0_dim[] = {3, 16, 32};
const nnom_qformat_param_t tensor_conv1d_3_kernel_0_dec[] = TENSOR_CONV1D_3_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_3_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_3_kernel_0 = {
    .p_data = (void*)tensor_conv1d_3_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_3_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_3_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_3_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_3_bias_0_data[] = TENSOR_CONV1D_3_BIAS_0;

const nnom_shape_data_t tensor_conv1d_3_bias_0_dim[] = {32};
const nnom_qformat_param_t tensor_conv1d_3_bias_0_dec[] = TENSOR_CONV1D_3_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_3_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_3_bias_0 = {
    .p_data = (void*)tensor_conv1d_3_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_3_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_3_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_3_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_3_output_shift[] = CONV1D_3_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv1d_3_bias_shift[] = CONV1D_3_BIAS_LSHIFT;
const nnom_conv2d_config_t conv1d_3_config = {
    .super = {.name = "conv1d_3"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_3_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_3_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_3_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_3_bias_shift, 
    .filter_size = 32,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_3_config = {
    .super = {.name = "max_pooling1d_3"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_conv1d_4_kernel_0_data[] = TENSOR_CONV1D_4_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_4_kernel_0_dim[] = {3, 32, 8};
const nnom_qformat_param_t tensor_conv1d_4_kernel_0_dec[] = TENSOR_CONV1D_4_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_4_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_4_kernel_0 = {
    .p_data = (void*)tensor_conv1d_4_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_4_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_4_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_4_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_4_bias_0_data[] = TENSOR_CONV1D_4_BIAS_0;

const nnom_shape_data_t tensor_conv1d_4_bias_0_dim[] = {8};
const nnom_qformat_param_t tensor_conv1d_4_bias_0_dec[] = TENSOR_CONV1D_4_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_4_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_4_bias_0 = {
    .p_data = (void*)tensor_conv1d_4_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_4_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_4_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_4_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_4_output_shift[] = CONV1D_4_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv1d_4_bias_shift[] = CONV1D_4_BIAS_LSHIFT;
const nnom_conv2d_config_t conv1d_4_config = {
    .super = {.name = "conv1d_4"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_4_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_4_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_4_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_4_bias_shift, 
    .filter_size = 8,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_4_config = {
    .super = {.name = "max_pooling1d_4"},
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

const nnom_shape_data_t tensor_dense_kernel_0_dim[] = {32, 6};
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

const nnom_shape_data_t tensor_dense_bias_0_dim[] = {6};
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
static int8_t nnom_output_data[6] = {0};

const nnom_shape_data_t tensor_output0_dim[] = {6};
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
	nnom_layer_t* layer[19];

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
	layer[10] = model.hook(conv2d_s(&conv1d_3_config), layer[9]);
	layer[11] = model.active(act_relu(), layer[10]);
	layer[12] = model.hook(maxpool_s(&max_pooling1d_3_config), layer[11]);
	layer[13] = model.hook(conv2d_s(&conv1d_4_config), layer[12]);
	layer[14] = model.active(act_relu(), layer[13]);
	layer[15] = model.hook(maxpool_s(&max_pooling1d_4_config), layer[14]);
	layer[16] = model.hook(flatten_s(&flatten_config), layer[15]);
	layer[17] = model.hook(dense_s(&dense_config), layer[16]);
	layer[18] = model.hook(output_s(&output0_config), layer[17]);
	model_compile(&model, layer[0], layer[18]);
	return &model;
}
