//*****************************************************************************
//
//! @file binary_counter.c
//!
//! @brief Example that displays the timer count on the LEDs.
//!
//! This example increments a variable on every timer interrupt. The global
//! variable is used to set the state of the LEDs. The example sleeps otherwise.
//!
//! SWO is configured in 1M baud, 8-n-1 mode.
//
//*****************************************************************************

//*****************************************************************************
//
// Copyright (c) 2019, Ambiq Micro
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
// 
// Third party software included in this distribution is subject to the
// additional license terms as defined in the /docs/licenses directory.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is part of revision 2.2.0 of the AmbiqSuite Development Package.
//
//*****************************************************************************
#define INPUT_SIZE_B sizeof(Input_data_s123[0])  // pass the int8 data to input, if it's other type, then sizeof(input_data)/sizeof(input_data[0])
#define INPUT_SIZE_S sizeof(Input_data_s3[0])
#define OUTPUT_SIZE_B sizeof(nnom_output_data_B)  // pass the int8 data to input, if it's other type, then sizeof(input_data)/sizeof(input_data[0])
#define OUTPUT_SIZE_S sizeof(nnom_output_data_S0)

#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "apollo2.h"

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include "nnom.h"
#include "multi_motions_s123_forNNoM_B.h"
#include "single_motion_0_forNNoM_S0.h"
#include "single_motion_1_forNNoM_S1.h"
#include "single_motion_2_forNNoM_S2.h"
#include "single_motion_3_forNNoM_S3.h"
#include "single_motion_4_forNNoM_S4.h"
#include "single_motion_5_forNNoM_S5.h"
#include "Apollo2_MOTION_Detector_s123_1IN.h"


/* Initialize variable */
int accurateCount = 0;
int B_invoked_times = 0;
int index_B = 0;
bool BIG_model_activate = true;
nnom_model_t* model;
static int8_t nnom_output_data_S[OUTPUT_SIZE_S] = {0};


void Infer_SMALL(int motion, int current_sample){
	/* Invoke the model */
	switch (motion){
		case 0:
			model = nnom_model_create_S0();
			memcpy(nnom_input_data_S0, Input_data_s3[current_sample], INPUT_SIZE_S);
			model_run(model);	
			memcpy(nnom_output_data_S, nnom_output_data_S0, OUTPUT_SIZE_S);
			break;
		case 1:
			model = nnom_model_create_S1();
			memcpy(nnom_input_data_S1, Input_data_s3[current_sample], INPUT_SIZE_S);
			model_run(model);	
			memcpy(nnom_output_data_S, nnom_output_data_S1, OUTPUT_SIZE_S);
			break;
		case 2:
			model = nnom_model_create_S2();
			memcpy(nnom_input_data_S2, Input_data_s3[current_sample], INPUT_SIZE_S);
			model_run(model);	
			memcpy(nnom_output_data_S, nnom_output_data_S2, OUTPUT_SIZE_S);
			break;
		case 3:
			model = nnom_model_create_S3();
			memcpy(nnom_input_data_S3, Input_data_s3[current_sample], INPUT_SIZE_S);
			model_run(model);	
			memcpy(nnom_output_data_S, nnom_output_data_S3, OUTPUT_SIZE_S);	
			break;
		case 4:
			model = nnom_model_create_S4();
			memcpy(nnom_input_data_S4, Input_data_s3[current_sample], INPUT_SIZE_S);
			model_run(model);	
			memcpy(nnom_output_data_S, nnom_output_data_S4, OUTPUT_SIZE_S);	
			break;
		case 5:
			model = nnom_model_create_S5();
			memcpy(nnom_input_data_S5, Input_data_s3[current_sample], INPUT_SIZE_S);
			model_run(model);	
			memcpy(nnom_output_data_S, nnom_output_data_S5, OUTPUT_SIZE_S);	
			break;
	}
	/* Print all the results */
//	am_util_stdio_printf("Model estimate = ");
//	for (int i = 0; i < OUTPUT_SIZE_S; i++){
//		am_util_stdio_printf("[%d] ", nnom_output_data_S[i]);
//	}
//	am_util_stdio_printf("\n");

	/* Find the result label with max probability */
	int maxVal = -128;
	int maxIndex = 0;
	for (int i = 0; i < OUTPUT_SIZE_S; i++){
			if (nnom_output_data_S[i] > maxVal) {
					maxVal = nnom_output_data_S[i];
					maxIndex = i;
			}
	}
	switch (maxIndex){
		case 0:
			BIG_model_activate = true;
//			am_util_stdio_printf("Output label changes, invoke BIG\n");
			break;
		case 1:
			BIG_model_activate = false;
//			am_util_stdio_printf("Output Label remains: [%d]\n", index_B);

			if (index_B == Output_expect[current_sample]){
				accurateCount++;
			}
			break;
	}
//	am_util_stdio_printf("Expect Label: %d\n", Output_expect[current_sample]);
	
	model_delete(model);
}

void Infer_BIG(int current_sample){
    /* Initialize model */
	model = nnom_model_create_B();
	/* Set input data */
	memcpy(nnom_input_data_B, Input_data_s123[current_sample], INPUT_SIZE_B);
	/* Invoke the model */
	model_run(model);

	/* Print all the results */
//	am_util_stdio_printf("Model estimate = ");
//	for (int i = 0; i < OUTPUT_SIZE_B; i++){
//		am_util_stdio_printf("[%d] ", nnom_output_data_B[i]);
//	}
//	am_util_stdio_printf("\n");

	/* Find the result label with max probability */
	int maxVal = -128;
	int maxIndex = 0;
	for (int i = 0; i < OUTPUT_SIZE_B; i++){
		if (nnom_output_data_B[i] > maxVal) {
			maxVal = nnom_output_data_B[i];
			maxIndex = i;
		}
	}
//	am_util_stdio_printf("Output Label: %d\n", maxIndex);
//	am_util_stdio_printf("Expect Label: %d\n", Output_expect[current_sample]);

	model_delete(model);
	
	index_B = maxIndex;
	BIG_model_activate = false;
	
	if (index_B == Output_expect[current_sample]){
		accurateCount++;
	}
}


int main(int argc, char* argv[]){
	// System Clock Setting
	am_hal_clkgen_sysclk_select(AM_HAL_CLKGEN_SYSCLK_MAX);
	// Set the default cache configuration
	am_hal_cachectrl_enable(&am_hal_cachectrl_defaults);
	// Low power setting
	am_bsp_low_power_init();
	// Print function set up
	am_util_stdio_printf_init((am_util_stdio_print_char_t) am_bsp_itm_string_print);
	am_bsp_pin_enable(ITM_SWO);
	am_hal_itm_enable();
	am_bsp_debug_printf_enable();
	// LED function set up
	am_devices_led_array_init(am_bsp_psLEDs, AM_BSP_NUM_LEDS);
	

	/* Program starts */
	am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS,15);
	am_util_delay_ms(2000);
	am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS,0);
	am_util_stdio_printf(" Datasets: %s\n\n", DATASETS_NAME);
	am_util_stdio_printf("=== Inferring ===\n\n");


	for (int current_sample = 0; current_sample < SAMPLE_COUNT; current_sample++) {  //the No. of input data
		if (!BIG_model_activate){
//			am_util_stdio_printf("\n[%dth] SMALL model is inferring:\n", index_B);
			Infer_SMALL(index_B, current_sample);
		}
		if (BIG_model_activate){
//			am_util_stdio_printf("BIG model is inferring:\n");
			Infer_BIG(current_sample);
			B_invoked_times++;
		}
	}
	
	
	am_util_stdio_printf("\nTest set accuracy = %d %%\n", (accurateCount*100)/SAMPLE_COUNT);
	am_util_stdio_printf("B invoked times: [%d]\n", B_invoked_times);
	am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS,15);
	am_util_delay_ms(500);
	am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS,0);		

	
// Go to deep sleep		
	while(1){	
		am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
	}
}
