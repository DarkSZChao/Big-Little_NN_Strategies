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

#include "ECM3532_MOTION_Detector_s123_1IN.h"


#define TASK_STACK_SIZE 2048
#define OUT_SIZE_S 2
#define OUT_SIZE_B 6
void infer_B(const q7_t *pIn2, q7_t *pOut0);  //q7_t is a ETA type
void infer_D(const q7_t *pIn0, q7_t *pOut0);  //q7_t is a ETA type


int accurateCount = 0;
int B_invoked_times = 0;
int index_B = 0;
bool BIG_model_activate = true;


void Infer_SMALL(int current_sample){
    q7_t pOut0[OUT_SIZE_S];

    q7_t Input_data_s3_dual[384*2];
    for (int i = 0; i < 384*2; i += 2){
        Input_data_s3_dual[i] = Input_data_s3[current_sample - 1][i/2];
        Input_data_s3_dual[i + 1] = Input_data_s3[current_sample][i/2];
    }

    //invoke the model
    infer_D(Input_data_s3_dual, pOut0);


    //print all the results
    ecm35xx_printf("Model estimate = ");
    for (int i = 0; i < OUT_SIZE_S; i++){
        ecm35xx_printf("[%d],", pOut0[i]);
    }
    ecm35xx_printf("\r\n");

    //find the result label with max probability
    int maxVal = 0;
    int maxIndex = 0;
    for (int i = 0; i < OUT_SIZE_S; i++){
        if (pOut0[i] > maxVal) {
            maxVal = pOut0[i];
            maxIndex = i;
        }
    }
    switch (maxIndex){
        case 0:
            BIG_model_activate = true;
            ecm35xx_printf("Output label changes, invoke BIG\r\n");
            break;
        case 1:
            BIG_model_activate = false;
            ecm35xx_printf("Output Label remains: [%d]\r\n", index_B);

            if (index_B == Output_expect[current_sample]){
                accurateCount++;
            }
            break;
    }
    ecm35xx_printf("Expect Label: [%d]\r\n\r\n", Output_expect[current_sample]);
}


void Infer_BIG(int current_sample){
    q7_t pOut0[OUT_SIZE_B];

    //invoke the model
    infer_B(Input_data_s123[current_sample], pOut0);

    //print all the results
    ecm35xx_printf("Model estimate = ");
    for (int i = 0; i < OUT_SIZE_B; i++){
        ecm35xx_printf("[%d],", pOut0[i]);
    }
    ecm35xx_printf("\r\n");

    //find the result label with max probability
    int maxVal = 0;
    int maxIndex = 0;
    for (int i = 0; i < OUT_SIZE_B; i++){
        if (pOut0[i] > maxVal) {
            maxVal = pOut0[i];
            maxIndex = i;
        }
    }
    ecm35xx_printf("Output Label: [%d]\r\n", maxIndex);
    ecm35xx_printf("Expect Label: [%d]\r\n\r\n", Output_expect[current_sample]);

    index_B = maxIndex;
    BIG_model_activate = false;

    if (index_B == Output_expect[current_sample]){
        accurateCount++;
    }
}


static void vInferTask(void *pvParameters){
    ecm35xx_printf(" Datasets: %s\r\n\r\n", DATASETS_NAME);
    uint64_t start_ms, stop_ms;
    size_t bmem, amem;
    ecm35xx_printf("=== Inferring ===\r\n");
    bmem = xPortGetFreeHeapSize();
    start_ms = HalTmrRead(0);

    for (int current_sample = 0; current_sample < SAMPLE_COUNT; current_sample++) {  //the No. of input data
        if (!BIG_model_activate){
            ecm35xx_printf("Dual SMALL model is inferring:\r\n");
            Infer_SMALL(current_sample);
        }
        if (BIG_model_activate){
            ecm35xx_printf("BIG model is inferring:\r\n");
            Infer_BIG(current_sample);
            B_invoked_times++;
        }
    }

    stop_ms =  HalTmrRead(0);
    amem = xPortGetFreeHeapSize();
    ecm35xx_printf("Test set accuracy = %d %%\r\n", (accurateCount*100)/SAMPLE_COUNT);
    ecm35xx_printf("B invoked times: [%d]\r\n\r\n", B_invoked_times);
    ecm35xx_printf("Heap In Bytes @ start %d, @ End %d Mem leakage %d\r\n", bmem, amem, (bmem - amem));
    ecm35xx_printf("Inference time= %d ms\r\n\r\n", (uint32_t) (stop_ms - start_ms));
}


int main( void ){

    ExecInit () ;
	xTaskCreate(vInferTask, "Executor_Compiler_Test", TASK_STACK_SIZE,
                NULL, tskIDLE_PRIORITY + 3, NULL);

	/* Start the scheduler. */
	vTaskStartScheduler();

	return 0;
}
