// Test data

// Name of datasets.
#define DATASETS_NAME "dataset"

// Number of datasets
#define SAMPLE_COUNT 10

// Datasets.
const int Input[SAMPLE_COUNT][784] = {
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,18,46,136,136,244,255,241,103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,94,163,253,253,253,253,238,218,204,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131,253,253,253,253,237,200,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,155,246,253,247,108,65,45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,207,253,253,230,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,157,253,253,125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,253,250,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,253,247,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,253,247,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,253,247,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,231,249,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,225,253,231,213,213,123,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,172,253,253,253,253,253,190,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,116,72,124,209,253,253,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,219,253,206,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,104,246,253,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,213,253,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,226,253,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,132,253,209,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,78,253,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,197,255,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,251,253,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,86,253,254,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,251,253,251,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,86,253,254,253,169,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,251,253,251,168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,86,253,254,253,169,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,196,253,251,168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,169,254,253,169,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,253,251,168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,169,254,253,169,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,253,251,168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,253,254,139,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,253,251,253,251,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,253,254,253,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,253,251,253,251,168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,198,253,254,253,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,251,253,251,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,253,254,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,83,196,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,238,217,68,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,150,254,254,254,232,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,224,254,145,254,240,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,160,253,254,254,187,254,180,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,184,254,184,207,254,254,248,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,252,206,17,47,254,254,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,254,43,0,165,254,159,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,237,23,42,235,218,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,213,254,105,212,247,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,249,254,254,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,254,254,163,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,164,254,243,254,89,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,56,249,177,49,235,202,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,213,249,50,0,212,247,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,69,254,158,0,0,212,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,212,254,30,0,0,212,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,235,213,8,0,37,243,241,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,240,200,4,5,193,254,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,232,254,212,218,254,195,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,68,216,254,254,166,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,113,207,253,255,253,143,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,219,252,252,252,253,252,252,234,146,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,191,252,239,180,55,196,214,252,252,252,57,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,101,176,65,0,0,0,28,199,252,252,253,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,205,252,253,167,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,253,255,253,196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,106,253,252,246,75,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,252,252,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,101,249,252,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,79,0,0,0,0,0,225,252,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,191,255,168,0,0,0,0,163,253,225,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,154,252,253,243,50,0,0,0,85,252,223,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,169,252,253,252,55,0,0,0,85,252,223,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,72,239,253,252,187,56,0,0,178,252,223,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,253,252,252,177,63,0,225,252,145,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,141,255,253,253,253,253,176,253,253,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,253,252,252,252,252,253,252,252,84,0,0,29,66,57,85,0,0,0,0,0,0,0,0,0,0,0,0,0,196,246,252,252,252,253,252,252,215,197,198,215,239,234,220,0,0,0,0,0,0,0,0,0,0,0,0,0,0,130,252,252,252,225,249,252,252,252,253,252,245,208,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,112,112,112,0,100,112,112,112,112,112,87,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,191,138,24,24,108,138,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,70,252,252,253,252,252,252,252,162,88,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,240,252,253,240,183,183,246,253,252,202,142,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37,98,211,206,0,0,42,109,177,252,252,211,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,18,0,0,0,0,5,54,179,252,220,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,241,255,92,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,230,253,92,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,68,246,247,67,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,134,252,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,248,200,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,97,222,192,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,99,208,227,174,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,207,252,237,88,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,202,253,244,207,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,96,252,252,244,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,199,249,253,128,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,118,248,253,113,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,115,253,240,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,253,252,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,253,231,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,170,255,254,219,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,238,254,252,245,253,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,238,254,230,75,0,162,195,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,173,254,215,13,0,20,211,218,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,196,240,30,0,0,64,254,249,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,240,157,0,5,111,245,254,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,254,124,97,191,254,254,254,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,208,254,254,254,254,254,247,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,246,248,147,100,254,199,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,39,0,46,254,118,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,254,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,173,244,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,221,186,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,42,254,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,118,254,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,147,254,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,209,211,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,209,155,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,209,125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,181,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,98,224,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,113,74,0,0,0,0,0,8,207,209,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,211,215,29,0,0,0,0,0,108,254,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,170,247,46,0,0,0,0,0,11,213,227,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,254,113,58,0,0,0,0,0,140,254,81,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,158,239,92,205,0,0,0,0,39,251,203,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,251,166,217,98,0,0,0,0,145,254,88,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,94,254,254,242,40,0,0,0,44,246,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,107,0,0,0,5,175,240,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,217,254,212,18,0,0,0,71,254,161,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,251,233,47,0,0,0,0,232,223,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,70,254,138,0,0,0,0,38,251,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,250,229,105,70,19,0,190,211,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,144,254,254,254,136,86,247,115,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,121,138,188,244,254,254,252,153,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,254,174,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,209,218,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,95,254,91,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,162,254,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,222,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,143,254,254,254,254,255,217,54,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,181,254,254,254,254,254,254,254,165,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,215,254,237,101,18,7,50,251,254,216,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,177,254,213,28,0,0,0,119,254,254,179,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,254,253,84,0,0,0,34,251,254,254,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,254,185,0,0,0,149,221,254,254,210,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,244,254,166,0,0,0,103,253,254,254,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,254,254,96,0,0,0,68,252,254,237,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,254,254,96,0,0,25,231,254,254,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,254,254,149,18,61,235,254,254,254,105,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,246,254,242,234,254,254,254,254,254,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,201,254,254,254,245,224,250,254,210,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,154,191,157,39,31,248,254,191,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,65,254,254,191,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,111,254,254,175,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,254,254,118,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,197,254,254,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,234,254,244,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,78,254,254,212,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,34,234,212,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,137,243,255,173,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,185,202,87,90,254,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,214,160,8,0,70,254,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,202,185,3,0,0,77,232,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,97,250,22,0,0,1,177,138,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,227,146,0,0,0,15,254,76,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,73,254,53,0,0,6,191,254,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,73,254,17,0,61,200,254,252,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,44,244,196,158,244,164,225,206,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,185,221,152,18,211,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,237,104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,246,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,155,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,155,174,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,161,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,235,99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,254,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100,246,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,194,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,165,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,146,254,255,251,95,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,97,234,254,254,232,254,254,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,140,254,254,174,67,33,200,254,190,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,108,253,254,235,51,1,0,0,12,254,253,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,216,254,244,55,0,0,0,0,6,213,254,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,254,254,132,0,0,0,0,0,0,168,254,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,254,243,34,0,0,0,0,0,0,168,254,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,254,157,0,0,0,0,0,0,0,168,254,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,228,254,105,0,0,0,0,0,0,7,228,254,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,58,254,254,87,0,0,0,0,0,0,10,254,246,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,58,254,254,9,0,0,0,0,0,0,10,254,210,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,58,254,254,9,0,0,0,0,0,0,105,254,91,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,219,254,9,0,0,0,0,0,24,230,254,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,216,254,9,0,0,0,0,0,84,254,251,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,216,254,36,0,0,0,0,22,208,251,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,120,0,0,0,3,140,254,229,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,254,222,17,0,0,91,254,236,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,235,254,134,21,119,237,254,124,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,53,249,254,234,252,254,172,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,237,254,254,133,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
};

//Labels: 10 categories 0 to 9.
const int Output[SAMPLE_COUNT] = {5,1,8,2,3,9,4,9,9,0};

