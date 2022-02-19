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

#include "ECM3532_MOTION_Detector.h"


#define TASK_STACK_SIZE 2048
#define OUT_SIZE 6
void infer(const q7_t *pIn0, q7_t *pOut0);  //q7_t is a ETA type


q7_t pOut0[OUT_SIZE];

static void vInferTask(void *pvParameters)
{
    ecm35xx_printf(" Datasets: %s\r\n\r\n", DATASETS_NAME);
    
    uint8_t accurateCount = 0;
    uint64_t start_ms, stop_ms;
    size_t bmem, amem;
    ecm35xx_printf("=== Inferring ===\r\n");
    bmem = xPortGetFreeHeapSize();
    start_ms = HalTmrRead(0);
    for (uint8_t current_sample = 0; current_sample < SAMPLE_COUNT; current_sample++) {  //the No. of input data
        //invoke the model
        infer(pIn0[current_sample], pOut0);

        //print all the results 
        ecm35xx_printf("Model estimate = ");
        for (uint8_t i = 0; i < OUT_SIZE; i++){
            ecm35xx_printf("[%d],", pOut0[i]);
        }
        ecm35xx_printf("\r\n");
                
        //find the result label with max probability
        uint8_t maxVal = 0;
        uint8_t maxIndex = 0;
        for (uint8_t i = 0; i < OUT_SIZE; i++){
            if (pOut0[i] > maxVal) {
                maxVal = pOut0[i];
                maxIndex = i;
            }
        }
        ecm35xx_printf("Output Label: %d\r\n", maxIndex);
        ecm35xx_printf("Expect Label: %d\r\n\r\n", pExpect[current_sample]);

        //print the accuracy
        if (maxIndex == pExpect[current_sample]){
            accurateCount++;
        }
    }
    stop_ms =  HalTmrRead(0);
    amem = xPortGetFreeHeapSize();
    ecm35xx_printf("Test set accuracy = %d %%\r\n\r\n", (accurateCount*100)/SAMPLE_COUNT);
    ecm35xx_printf("Heap In Bytes @ start %d, @ End %d Mem leakage %d\r\n", bmem, amem, (bmem - amem));
    ecm35xx_printf("Inference time= %d ms\r\n\r\n", (uint32_t) (stop_ms - start_ms));
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
