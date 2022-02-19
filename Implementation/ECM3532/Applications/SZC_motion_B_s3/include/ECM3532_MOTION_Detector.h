// Test data.

// Name of datasets.
#define DATASETS_NAME "dataset"

// Number of datasets
#define SAMPLE_COUNT 10

// Datasets.
const q7_t pIn0[SAMPLE_COUNT][384] = {
{-31,0,31,-36,-4,33,-19,1,28,-19,11,27,-9,7,27,4,6,27,22,3,27,61,0,27,68,5,31,51,3,36,47,-10,40,49,-36,26,45,-48,5,14,-32,7,-19,-17,14,-22,-10,12,-8,-4,12,4,-6,9,17,-10,3,19,-12,6,2,-21,9,0,-30,9,-4,-25,11,-19,-16,11,-16,-8,11,-13,2,10,1,10,4,12,14,2,3,16,5,4,12,8,2,7,9,4,2,7,15,-6,5,16,-10,4,21,-7,7,23,-3,12,25,4,11,33,12,9,39,10,2,42,-3,-5,27,-22,-4,6,-40,-1,5,-44,-5,24,-40,-10,50,-36,-12,62,-29,-8,52,-15,3,28,2,8,2,15,13,-10,13,16,-16,4,10,-24,1,16,-22,-3,22,-19,-9,23,-25,-8,31,-21,-6,30,-19,0,28,-17,7,29,-3,6,22,5,2,20,18,-4,21,39,-4,21,51,1,24,56,3,29,50,1,36,46,-11,32,50,-35,19,41,-50,6,13,-35,3,-25,-9,13,-30,-3,15,4,-3,9,26,-7,6,26,-17,6,13,-23,11,-6,-25,15,-15,-15,13,-20,1,11,-18,6,9,-6,8,7,6,11,5,11,15,7,2,15,11,-3,13,12,2,15,10,5,8,6,13,-2,2,18,-8,2,16,-9,5,19,-5,8,32,-6,10,45,-4,11,44,-4,3,36,-10,-7,28,-16,-4,11,-29,-4,16,-41,-15,37,-41,-14,43,-38,-2,45,-24,9,33,-2,10,9,12,6,0,13,8,-10,6,9,-22,1,10,-27,-5,22,-32,-10,30,-24,-8,31,-18,-5,29,-18,2,26,-13,7,27,-10,6,26,1,1,22,19,-3,26,46,-1,27,72,3,25,74,2,23,61,-8,22,55,-34,19,49,-54,13,24,-35,13,-10,-12,15,-15,-11,16,-12,2,17,0,6,6,35,-18,-5,27,-29,3,-5,-32,17},
{11,-14,11,11,-14,10,11,-14,11,11,-14,11,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,11,11,-14,11,11,-14,11,11,-14,10,11,-14,10,10,-14,10,11,-14,10,11,-14,11,11,-14,11,11,-14,10,11,-13,11,11,-13,11,11,-13,11,11,-13,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-13,11,11,-14,11,11,-14,11,11,-14,10,11,-14,10,10,-15,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,10,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,11,11,-14,11,11,-14,10,11,-14,10,11,-14,10,11,-14,11,11,-14,11,11,-14,11,11,-14,10,11,-14,10,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,10,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,10,11,-14,10,11,-14,10,11,-13,10,11,-14,10,11,-14,10,11,-14,10,11,-14,11,11,-13,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,11,11,-14,10,11,-14,10,11,-14,11,11,-14,11,11,-14,11,11,-14,10,11,-14,10,11,-14,10,11,-14,10,11,-14,11,11,-14,10,11,-14,10,11,-14,10,11,-14,10},
{-10,-11,15,-3,-21,8,20,-31,-9,61,-38,-26,77,-38,-25,69,-18,-5,67,5,6,75,13,-7,72,9,-19,48,7,-7,18,10,2,4,5,-5,5,-2,-3,-7,-3,-1,-16,-5,-5,-7,-3,-4,-9,-1,-9,-11,3,-10,-11,7,-4,-21,4,1,-12,5,6,5,8,5,12,7,-1,8,8,-2,-7,5,-1,-2,3,2,10,5,1,13,6,0,0,2,-2,-21,-9,-7,-7,-36,-5,25,-73,2,45,-80,6,59,-57,8,51,-29,3,49,14,0,60,22,2,62,-18,1,75,-25,-2,72,-14,-4,42,-9,-2,18,8,3,-10,10,5,-30,9,9,-30,7,18,-16,-11,23,8,-17,12,18,-24,-2,18,-29,6,6,-20,20,-8,-15,24,3,-8,30,13,-1,26,11,3,20,9,5,20,6,1,17,10,0,18,14,-3,16,9,-4,18,6,-3,27,11,-9,26,29,-21,19,54,-37,5,55,-40,-13,22,-25,1,-11,-10,20,-14,-3,16,0,-4,3,6,-6,-10,4,-13,-9,9,-28,1,15,-34,0,10,-20,2,8,-6,6,9,-2,1,6,-3,-1,1,-3,-1,2,-3,-2,7,-1,-2,11,0,-7,13,-2,-7,8,-4,-3,-4,-2,1,-16,1,4,-21,3,1,-16,-3,0,-12,-11,2,-10,-9,6,-7,0,13,-11,6,10,-11,8,2,-2,1,-5,3,-6,-7,-4,-6,-4,-16,-12,5,-13,-24,20,9,-30,26,53,-27,13,98,-23,7,96,-33,16,67,-39,20,53,-22,15,45,0,7,38,10,2,23,10,6,-5,6,19,-27,2,29,-25,2,25,-3,0,16,0,-7,13,-9,-7,15,-21,-3,19,-46,-3,22,-44,1,19,-24,2,10,-8,-4,2,17,-12,0,14,-16,6,-2,-14,14,0,-18,13,7,-29,4,26,-33,-11,41,-27,-20,48,-16,-17,57,-11,-12,64,-9,-9,74,-4,-7,55,1,-3},
{14,24,21,13,25,21,13,24,21,13,24,21,13,24,22,13,24,22,13,24,22,13,24,21,14,24,21,14,24,21,13,24,21,13,24,22,13,24,22,13,24,22,13,24,21,13,24,20,14,24,20,14,24,21,13,24,21,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,21,13,24,21,13,24,22,13,24,22,13,24,22,14,24,22,13,24,22,13,24,22,13,24,22,13,25,22,13,25,22,13,25,22,13,25,22,13,25,21,13,25,21,13,25,22,13,24,22,14,24,21,13,24,21,13,25,21,13,25,22,13,25,22,13,25,21,13,25,21,13,24,21,13,24,21,13,24,21,13,24,22,13,24,22,13,24,21,13,24,21,13,24,21,13,25,21,13,24,21,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,21,13,24,21,13,24,21,13,25,21,13,24,21,13,24,21,13,25,21,13,25,21,13,24,21,13,24,22,13,24,22,13,25,22,13,24,22,13,24,22,13,24,22,13,24,21,13,24,21,13,24,21,13,24,21,13,24,21,14,25,21,14,25,21,14,24,22,13,25,22,13,25,23,13,25,22,13,25,21,13,25,21,13,24,21,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,13,24,22,14,24,22,14,24,22,14,24,22,14,24,22,13,24,22,13,24,22,13,25,22,13,25,22,13,24,22,13,24,21,13,24,21,13,24,21,13,24,21,13,24,21,13,24,21,13,25,21,13,25,21,13,24,21,13,24,21,13,24,21,13,24,21,13,25,22,14,25,22},
{22,-62,6,7,-21,-3,2,-16,-13,5,-27,-14,13,-37,-8,17,-45,-2,10,-42,1,2,-28,1,-5,-18,-2,-6,-11,-3,-9,-3,-7,-16,0,-9,-15,-5,-6,-15,-13,-6,-16,-20,-4,-13,-21,-3,-11,-17,-5,-6,-7,-4,-1,2,-4,1,4,-4,5,-2,-2,7,-7,-2,8,-6,0,9,-5,2,10,-4,3,10,-3,6,9,-3,8,8,3,13,10,6,12,16,3,5,26,-2,1,29,-8,0,24,-11,3,23,-17,10,25,-23,11,28,-17,11,35,-13,9,43,-26,4,47,-49,-7,48,-68,-14,34,-66,-1,15,-50,15,13,-41,16,19,-28,13,14,-18,10,7,-22,4,1,-27,-1,-5,-38,-4,-12,-43,-5,-24,-32,-3,-30,-23,-2,-22,-13,0,-17,-4,5,-12,3,13,-7,9,17,-4,3,17,2,-3,17,2,-6,17,3,-12,17,10,-15,17,12,-18,18,13,-16,20,10,-12,26,5,-8,33,9,-4,32,26,-11,17,52,-26,3,64,-35,5,50,-36,9,25,-33,1,0,-35,-3,5,-36,-4,27,-34,-8,25,-35,-11,25,-34,-12,24,-25,-7,12,-16,-2,15,-13,-4,10,-15,-3,-6,-9,0,-12,-6,1,-19,-7,1,-28,0,0,-36,3,1,-30,0,0,-18,0,-4,-16,-4,-8,-7,-12,-9,-3,-14,-6,-2,-10,-3,8,-3,-2,10,5,-2,17,11,-3,22,9,-3,16,5,-3,19,0,-4,26,-8,-3,35,-15,-4,37,-25,-5,35,-36,-7,45,-39,-10,55,-40,-10,56,-43,-10,40,-43,-7,17,-37,-3,15,-32,0,24,-27,2,24,-20,5,20,-19,7,14,-20,5,5,-22,3,-7,-24,3,-21,-22,4,-29,-17,9,-30,-7,10,-31,-2,11,-32,-1,15,-30,3,16,-26,4,16,-24,2,17,-17,-1,17,-6,-8,18,0,-15,19,5,-14,23,14,-7,26,17,-2,24,20,1,22,34,0,19},
{-64,95,54,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,52,-64,95,52,-64,94,53,-64,94,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,54,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,52,-64,95,52,-64,95,52,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,94,53,-64,94,53,-63,95,53,-64,95,53,-64,95,54,-64,95,54,-64,95,54,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-63,94,53,-64,94,53,-64,94,53,-65,94,53,-64,94,53,-64,94,53,-64,94,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,52,-64,95,52,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,94,53,-64,94,53,-64,94,54,-64,94,54,-64,95,53,-64,95,53,-64,95,53,-63,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-65,95,53,-65,95,53,-64,95,53,-64,95,54,-64,95,54,-65,95,54,-64,95,54,-64,95,53,-64,95,53,-64,95,53,-64,95,54,-64,95,53,-64,94,53,-64,94,53,-64,94,53,-64,95,53,-64,95,53,-64,95,53,-64,94,53,-64,95,53,-65,96,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,52,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,52,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,54,-64,95,54,-64,95,53,-64,95,52,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,95,53,-64,94,53,-63,94,53,-63,94,53,-64,94,53,-64,94,53,-64,95,53},
{13,18,32,13,17,32,13,16,31,13,16,31,13,17,31,13,16,31,13,16,31,12,16,31,12,15,31,13,15,31,13,15,30,13,15,30,13,15,30,13,15,30,13,15,31,13,15,31,13,15,30,13,16,30,13,16,30,13,16,29,13,15,30,13,15,30,14,15,30,14,15,29,14,15,30,13,16,30,13,16,30,13,16,30,13,16,30,13,15,30,13,15,29,14,15,29,14,15,29,13,15,29,13,15,29,13,15,29,13,15,29,13,16,29,13,16,29,13,16,29,13,15,29,13,15,29,13,16,29,13,16,29,13,16,29,13,15,29,13,15,29,13,15,30,13,15,30,13,15,30,13,15,30,13,15,29,13,15,29,13,15,29,13,15,28,13,16,29,13,15,30,13,15,29,13,15,30,13,16,29,13,16,29,13,16,29,13,16,29,13,15,28,13,15,29,13,15,29,13,15,29,13,15,29,13,15,28,13,15,28,13,15,28,13,15,28,13,15,29,13,15,29,13,15,29,13,15,29,13,15,29,13,15,29,13,15,29,13,15,29,13,15,29,13,15,29,13,15,29,13,15,29,13,15,28,13,15,28,13,15,29,14,15,29,13,15,30,13,15,30,13,15,30,13,15,30,14,15,30,14,15,30,13,15,30,13,15,30,12,15,31,12,15,31,14,15,28,15,14,26,14,14,26,13,14,28,12,14,29,13,14,29,13,16,29,13,17,28,13,16,27,13,14,26,13,15,28,13,16,28,13,16,28,13,17,28,13,17,28,13,17,28,13,16,28,13,15,28,13,15,28,13,16,28,13,15,29,13,16,29,13,16,29,13,16,29,14,16,29,13,15,29,13,15,29,13,15,29,13,14,29,13,15,28},
{12,21,5,7,20,9,6,12,15,11,-7,20,22,-21,22,49,-18,23,84,2,24,98,29,24,86,47,22,72,45,20,61,35,25,40,31,30,25,30,29,18,31,26,1,28,24,-14,21,21,-25,17,19,-28,11,19,-23,8,18,-23,8,19,-22,7,19,-20,9,18,-15,12,18,-6,17,17,-2,25,19,-5,29,23,-5,33,22,-4,35,22,-5,33,23,-4,28,23,-1,24,24,2,23,23,5,21,21,19,5,16,51,-20,7,80,-36,-6,81,-31,-8,72,-6,12,67,30,34,61,47,32,56,34,16,52,25,6,50,25,4,47,17,6,36,17,6,29,18,4,18,18,6,-8,22,6,-17,19,5,-5,16,5,-7,13,1,-7,7,1,-4,9,2,-9,7,3,-11,6,5,-16,11,6,-20,9,5,-14,13,5,-8,19,5,-3,23,7,-3,26,8,-11,26,11,-14,24,15,-11,20,15,-3,14,14,9,11,12,14,7,13,20,-1,16,31,-11,21,47,-13,24,75,-10,25,103,-4,29,107,5,31,93,6,30,73,13,26,45,29,23,31,23,28,29,14,32,8,23,29,-2,23,24,-8,23,21,-25,27,20,-21,18,21,-22,11,23,-27,9,24,-19,10,25,-24,17,25,-24,20,24,-21,26,23,-23,27,20,-13,21,19,-8,17,20,1,14,17,14,18,16,15,25,15,16,27,14,10,28,14,0,22,11,-1,11,12,17,-3,12,55,-17,1,87,-18,-16,99,-6,-27,94,9,-19,77,18,2,46,26,25,7,37,34,-5,42,21,6,41,13,15,32,3,23,13,-18,14,4,-10,-10,7,8,-22,11,5,-21,15,6,-8,20,5,10,20,-3,12,17,0,1,18,3,-7,20,7,-9,20,10,-5,22,6,-6,17,9,-12,14,11,-18,16,12,-19,16,13,-8,16,8,3,15,6},
{-82,102,42,-82,102,43,-82,102,42,-82,103,42,-82,102,42,-82,102,42,-82,103,42,-82,103,42,-81,103,42,-81,103,42,-81,103,42,-81,103,42,-81,103,42,-81,102,42,-82,102,42,-82,102,42,-82,102,42,-82,102,43,-82,103,43,-81,103,43,-81,103,43,-82,103,43,-82,103,42,-82,102,42,-82,102,42,-82,102,42,-82,102,42,-81,102,42,-81,102,42,-81,101,42,-81,102,42,-82,102,43,-82,102,43,-82,102,42,-81,102,41,-81,103,42,-82,103,43,-82,103,43,-82,102,42,-82,103,42,-82,103,42,-82,103,42,-82,103,42,-82,103,42,-81,103,42,-81,102,42,-81,102,42,-81,102,42,-81,102,42,-81,102,42,-81,102,42,-82,103,42,-82,103,42,-81,103,42,-82,102,42,-82,102,43,-82,103,43,-82,103,42,-82,103,42,-82,103,42,-81,102,42,-81,103,43,-82,103,43,-82,102,42,-82,102,42,-81,102,42,-81,102,42,-81,102,42,-81,102,42,-81,102,43,-82,102,43,-82,102,42,-81,102,41,-81,103,41,-81,103,42,-82,103,42,-81,102,42,-82,102,42,-82,103,42,-82,102,41,-81,103,41,-81,102,42,-81,102,42,-81,102,41,-81,102,41,-81,102,41,-81,102,41,-81,103,41,-81,103,41,-81,103,42,-82,103,42,-82,103,42,-81,102,42,-81,102,42,-82,103,41,-82,102,41,-82,102,41,-82,102,42,-82,102,42,-82,102,42,-82,102,42,-82,102,42,-82,102,41,-81,102,41,-81,103,42,-82,103,42,-82,103,42,-81,102,42,-81,103,42,-82,103,42,-82,103,41,-82,103,41,-82,103,41,-82,102,41,-82,102,41,-82,102,41,-82,102,42,-82,103,42,-82,103,42,-82,102,42,-81,102,42,-81,102,42,-81,102,42,-81,103,42,-81,103,42,-81,103,42,-82,103,42,-82,102,42},
{13,0,26,13,1,26,13,0,26,13,1,26,13,1,26,13,0,25,13,0,26,13,0,26,13,0,26,13,0,27,13,1,26,13,1,26,13,1,27,13,1,27,13,0,27,13,0,27,13,1,27,13,1,27,13,0,26,13,0,26,13,0,26,13,1,26,13,1,26,13,1,26,13,1,26,13,1,26,13,1,26,13,1,25,13,1,25,13,1,25,13,0,25,13,0,25,13,0,26,13,0,26,13,0,25,13,0,26,13,0,26,13,0,26,13,0,26,13,1,26,13,0,25,13,0,25,13,0,25,13,0,25,13,0,25,13,0,25,13,0,26,13,0,26,13,0,26,13,0,25,13,0,25,13,0,26,14,0,26,13,0,26,13,1,26,13,1,26,13,0,25,13,0,25,13,0,25,13,0,26,13,0,26,13,0,26,13,0,26,13,0,25,13,1,25,13,1,25,13,1,25,13,0,25,13,0,25,13,1,25,13,1,25,13,1,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,1,26,13,0,26,13,0,26,13,0,26,13,1,26,13,1,26,13,1,26,13,1,25,13,0,25,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,12,0,27,13,0,27,13,0,27,13,0,26,13,0,26,13,1,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26,13,1,26,13,1,26,13,1,26,13,1,26,13,0,26,13,0,26,13,0,26,13,0,26,13,0,27,13,0,26,13,0,26,13,0,26,13,0,26,13,0,26}
};

//Labels: 6 categories 0 to 5.
const q7_t pExpect[SAMPLE_COUNT] = {0,4,2,3,1,5,4,2,5,4};
