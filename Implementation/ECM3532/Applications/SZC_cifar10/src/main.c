/*******************************************************************************
 *
 * Copyright (C) 2019 Eta Compute, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/
#include "config.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "gpio_hal.h"
#include "timer_hal.h"
#include "print_util.h"
#include <stdint.h>
#ifdef CONFIG_CLI
#include "FreeRTOS_CLI.h"
#include "UARTCommandConsole.h"
#endif
#include "executor_public.h"
#include "arm_math.h"
#include "eta_types.h"

#include "input_horse.h"
#include "input_car.h"
#include "input_deer.h"

#define TASK_STACK_SIZE 2048
#define OUT_SIZE 10
void infer(const q7_t *pIn0, q7_t *pOut0);  //q7_t is a ETA type

const char labels[OUT_SIZE][12] = {
    "AIRPLANE", "AUTOMOBILE", "BIRD", "CAT", "DEER",
    "DOG", "FROG", "HORSE", "SHIP", "TRUCK"
};
#define input_total_No 3
const q7_t *Input_list[input_total_No] = {pIn0_car, pIn0_deer, pIn0_horse};
const q7_t *Expect_list[input_total_No] = {pExpect_car, pExpect_deer, pExpect_horse};
q7_t pOut0[OUT_SIZE];


static void vInferTask(void *pvParameters)
{
    uint8_t maxVal, maxIndex;
    uint64_t start_ms, stop_ms;
    size_t bmem, amem;
    for (uint8_t input_No = 0; input_No < input_total_No; ++input_No) {  //the No. of input data
        bmem = xPortGetFreeHeapSize();
        ecm35xx_printf("=== Inferencing ===\r\n");
        start_ms = HalTmrRead(0);
        infer(Input_list[input_No], pOut0);  //invoke the model
        stop_ms =  HalTmrRead(0);

        //print all the results 
        for (uint8_t i = 0; i < OUT_SIZE; ++i){
            ecm35xx_printf(" pOut0[%d] = %d,\tpExpect[%d] = %d\r\n", i, pOut0[i], i, Expect_list[input_No][i]);
        }
        
        //find the result label with max probability
        maxVal = 0; maxIndex = 100;
        for (uint8_t i = 0; i < OUT_SIZE; ++i){
            if (pOut0[i] > maxVal) {
                maxVal = pOut0[i];
                maxIndex = i;
            }
        }
        ecm35xx_printf("Output: %s\r\n", labels[maxIndex]);
        //find the expected label with max probability
        maxVal = 0; maxIndex = 100;
        for (uint8_t i = 0; i < OUT_SIZE; ++i){
            if (Expect_list[input_No][i] > maxVal) {
                maxVal = Expect_list[input_No][i];
                maxIndex = i;
            }
        }
        ecm35xx_printf("Expect: %s\r\n", labels[maxIndex]);

        amem = xPortGetFreeHeapSize();
        ecm35xx_printf("Heap In Bytes @ start %d, @ End %d Mem leakage %d\r\n", bmem, amem, (bmem - amem));
        ecm35xx_printf("Inference time= %d ms\r\n\r\n", (uint32_t) (stop_ms - start_ms));
    }

}


int main( void )
{

    ExecInit () ;
	xTaskCreate(vInferTask, "Executor_Compiler_Test", TASK_STACK_SIZE,
                NULL, tskIDLE_PRIORITY + 3, NULL);

	/* Start the scheduler. */
	vTaskStartScheduler();

	return 0;
}
