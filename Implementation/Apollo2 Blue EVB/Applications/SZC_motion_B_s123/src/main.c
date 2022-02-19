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
#define INPUT_SIZE sizeof(nnom_input_data)
#define OUTPUT_SIZE sizeof(nnom_output_data)

#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "apollo2.h"

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include "nnom.h"
#include "multi_motions_s123_forNNoM.h"
#include "NNoM_Apo_STM_MOTION_Detector_s123.h"


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
	am_util_delay_ms(500);
	am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS,0);		
	am_util_stdio_printf("=== Inferring ===\n\n");


  /* Initialize model */
	nnom_model_t* model;
	model = nnom_model_create();
	int accurateCount = 0;
	
	for(int current_sample = 0; current_sample < SAMPLE_COUNT; current_sample++){
		/* Set input data */
		memcpy(nnom_input_data, Input_data[current_sample], INPUT_SIZE);
		/* Invoke the model */
		model_run(model);
		
		/* Print all the results */
		am_util_stdio_printf("\nModel estimate = ");
		for (int i = 0; i < OUTPUT_SIZE; i++){
			am_util_stdio_printf("[%d] ", nnom_output_data[i]);
		}
		am_util_stdio_printf("\n");
		
		/* Find the result label with max probability */
		int maxVal = -128;
		int maxIndex = 0;
		for (int i = 0; i < OUTPUT_SIZE; i++){
            if (nnom_output_data[i] > maxVal) {
                    maxVal = nnom_output_data[i];
                    maxIndex = i;
            }
		}
		am_util_stdio_printf("Output Label: %d\n", maxIndex);
		am_util_stdio_printf("Expect Label: %d\n", Output_expect[current_sample]);

		/* Print the accuracy */
		if (maxIndex == Output_expect[current_sample]){
            accurateCount++;
		}
	}

	model_delete(model);

	
	am_util_stdio_printf("\nTest set accuracy = %d %%\n", (accurateCount*100)/SAMPLE_COUNT);
	am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS,15);
	am_util_delay_ms(500);
	am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS,0);		

	
// Go to deep sleep		
	while(1){	
		am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
	}
}
