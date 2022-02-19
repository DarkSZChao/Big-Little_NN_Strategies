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
#include "uart_hal.h"
#include "print_util.h"
#include <stdint.h>
#include "semphr.h"
#include "eta_chip.h"
#ifdef CONFIG_CLI
#include "FreeRTOS_CLI.h"
#include "UARTCommandConsole.h"
#endif
#include "executor_public.h"
#include "arm_math.h"
#include "eta_types.h"

#include "ECM3532_MOTION_Detector_s123_1IN.h"




#ifdef CONFIG_CAM_HM01B0
#include "eta_devices_hm01b0.h"
#elif defined CONFIG_CAM_HM0360
#include "eta_devices_hm0360.h"
#endif

#ifdef CONFIG_DISPLAY_NHD0216
#include "eta_devices_nhd0216.h"
#endif

#define TASK_STACK_SIZE  (256)
#define MAX_DETECTIONS (100)

#ifdef CONFIG_DISPLAY_NHD0216
tNhd0216Cfg sCfg;
#endif

#ifdef CONFIG_CAM_HM01B0
#define D0_PIN CONFIG_HM01B0_D0_PIN
#define PCLK_PIN CONFIG_HM01B0_PCLK_GPIO
#define HSYNC_PIN CONFIG_HM01B0_HSYNC_GPIO
#define VSYNC_PIN CONFIG_HM01B0_VSYNC_GPIO
#elif defined CONFIG_CAM_HM0360
#define D0_PIN CONFIG_HM0360_D0_PIN
#define PCLK_PIN CONFIG_HM0360_PCLK_GPIO
#define HSYNC_PIN CONFIG_HM0360_HSYNC_GPIO
#define VSYNC_PIN CONFIG_HM0360_VSYNC_GPIO
#endif

// this needs to go in config.h manually. No Kconfig support
// only stall if delay is greater than 1000
#define configEXPECTED_IDLE_TIME_BEFORE_SLEEP 1001
// PIR Motion detect useses interrupt. Comment out to use polling
#include "semphr.h"
#include "eta_chip.h"

// GPIO Pin Definitions
#define PIR_PIN 5
uint32_t pir_argv = PIR_PIN;
#define PUMPKIN_PIN 27
#define LED_RED 4
#define LED_BLUE 28
#define SPI_FLASH_POWER 11
#define PDM_MIC_POWER 2
#define CAM_ALS_ACCL_INT 1
#define CAM_MCLK_ENA 6
#define CAM_XSLEEP 0
// selects either GPIO CHA or GPIO CHB for interupt
#define GPIO_AB_SEL 1

#define LEDON 1
#define IFLEDON if(LEDON)




//#define TASK_STACK_SIZE 2048
#define OUT_SIZE_S 2
#define OUT_SIZE_B 6
void infer_B(const q7_t *pIn2, q7_t *pOut0);  //q7_t is a ETA type
void infer_S0(const q7_t *pIn0, q7_t *pOut0);  //q7_t is a ETA type
void infer_S1(const q7_t *pIn0, q7_t *pOut0);  //q7_t is a ETA type
void infer_S2(const q7_t *pIn0, q7_t *pOut0);  //q7_t is a ETA type
void infer_S3(const q7_t *pIn0, q7_t *pOut0);  //q7_t is a ETA type
void infer_S4(const q7_t *pIn0, q7_t *pOut0);  //q7_t is a ETA type
void infer_S5(const q7_t *pIn0, q7_t *pOut0);  //q7_t is a ETA type



// Interrupt for PIR Motion on GPIO5
SemaphoreHandle_t PirMotionSemaphore = NULL;
uint8_t one_intr = 0;
/* pir gpio isr*/
void pir_wake_up_isr(void *data)
{
	static signed portBASE_TYPE xHigherPriorityTaskWoken;

	// disable the interrupt until after inference
	HalGpioIntDisable(GPIO_AB_SEL, PIR_PIN);
	HalGpioIntClear(GPIO_AB_SEL, PIR_PIN);

	xHigherPriorityTaskWoken = pdFALSE;

	// others are using GPIO 0/1 so must also check for correct pin
	if((*((uint32_t *)data) == PIR_PIN) && (PirMotionSemaphore != NULL))
	{
		// set flag indicating interupt happened.
		// debug since interupts are getting enabled before end of infer loop
		one_intr=1;
		xSemaphoreGiveFromISR( PirMotionSemaphore, &xHigherPriorityTaskWoken );
	}
	
	if( xHigherPriorityTaskWoken != pdFALSE )
	{
		portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
	}
}


int accurateCount = 0;
int index_B = 0;
bool BIG_model_activate = true;


void Infer_SMALL(int motion, int current_sample){
    q7_t pOut0[OUT_SIZE_S];

    //invoke the model
    switch (motion){
        case 0:
            infer_S0(Input_data_s3[current_sample], pOut0);
            break;
        case 1:
            infer_S1(Input_data_s3[current_sample], pOut0);
            break;
        case 2:
            infer_S2(Input_data_s3[current_sample], pOut0);
            break;
        case 3:
            infer_S3(Input_data_s3[current_sample], pOut0);
            break;
        case 4:
            infer_S4(Input_data_s3[current_sample], pOut0);
            break;
        case 5:
            infer_S5(Input_data_s3[current_sample], pOut0);
            break;
    }

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

    for (int current_sample = 0; current_sample < 1; current_sample++) {  //the No. of input data
        if (!BIG_model_activate){
            ecm35xx_printf("[%dth] SMALL model is inferring:\r\n", index_B);
            Infer_SMALL(index_B, current_sample);
        }
        if (BIG_model_activate){
            ecm35xx_printf("BIG model is inferring:\r\n");
            Infer_BIG(current_sample);
        }
    }

    stop_ms =  HalTmrRead(0);
    amem = xPortGetFreeHeapSize();
    ecm35xx_printf("Test set accuracy = %d %%\r\n\r\n", (accurateCount*100)/SAMPLE_COUNT);
    ecm35xx_printf("Heap In Bytes @ start %d, @ End %d Mem leakage %d\r\n", bmem, amem, (bmem - amem));
    ecm35xx_printf("Inference time= %d ms\r\n\r\n", (uint32_t) (stop_ms - start_ms));
    
    HalGpioIntInit(PIR_PIN, GPIO_AB_SEL, pir_wake_up_isr, &pir_argv, HalGpioTrigHigh, HalGpioPullDown);
}




// vTaskDelay consumes a lot of power
// try other delay types
void LocalTaskDelay(uint64_t ui64DelayMs)
{
#if 0
	vTaskDelay((uint32_t) ui64DelayMs);
#else
	uint64_t ui64Count = EtaCspTimerCountGetMs();
	uint64_t ui64Expire = ui64Count + ui64DelayMs;

	//
	// Poll the timer until the desired number of milliseconds has been reached.
	//
	do
	{
		ui64Count = EtaCspTimerCountGetMs();
	}
	while(ui64Count < ui64Expire);
#endif
}


static void vLedBlinkTask(void *pvParameters)
{
	HalGpioOutInit(LED_RED, 0);
	HalGpioOutInit(LED_BLUE, 1);

	while (1) {
		ecm35xx_printf("vLedBlinkTask\r\n");
		HalGpioToggle(LED_RED);
		HalGpioToggle(LED_BLUE);
		LocalTaskDelay(1000);
	}
}

static void vTinyeyeTask(void *pvParameters)
{
	q7_t *conf, *loc;
	size_t bmem, amem;
	uint32_t ui32Cnt = 0, images;
	uint8_t UartBuffer;

	// a pause after power up before running just for debug
	ecm35xx_printf("Startup Delay.\r\n");
	IFLEDON LocalTaskDelay(2000);
	// signal code is starting
	IFLEDON HalGpioOutInit(LED_RED,1);
	IFLEDON HalGpioOutInit(LED_BLUE,1);
	LocalTaskDelay(500);
	ecm35xx_printf("Running Detection\r\n");

	// enable motion detect after startup delays
	PirMotionSemaphore = xSemaphoreCreateBinary();
	if(PirMotionSemaphore == NULL)
	{
		configASSERT(0);
	}
	HalGpioIntInit(PIR_PIN, GPIO_AB_SEL, pir_wake_up_isr, &pir_argv, HalGpioTrigHigh, HalGpioPullDown);

   while(1)
   {

		// bmem = xPortGetFreeHeapSize();
		//ecm35xx_printf("=== Inferencing loop cnt [%d] ===\r\n", ui32Cnt++);

		// wait and stall in xSemaphoreTake until there is motion
		ecm35xx_printf("Waiting for motion\r\n");
		// if((Hm0360RegRead(0x100)&&0x7)==0x0) HalGpioOutInit(LED_BLUE,0);

		// put camera to sleep
		HalGpioWrite(CAM_XSLEEP,0);
		// turn off camera mclk to save power
		// worth 200uA at 3.3V
		HalGpioWrite(CAM_MCLK_ENA,1);
		ecm35xx_printf("Semaphore \r\n");
		ecm35xx_printf("Semaphore %x \r\n", portMAX_DELAY);
		
		if( xSemaphoreTake( PirMotionSemaphore, 1000) == pdTRUE ) {
		ecm35xx_printf("Motion  detected\r\n");
		}	
		else
		{
		ecm35xx_printf("Motion not detected\r\n");
		}	
		LocalTaskDelay(2000);
		
		if( xSemaphoreTake( PirMotionSemaphore, portMAX_DELAY) == pdTRUE ) {
			// just a flag to indicate if PIR interupt happened
			// clear it here and check it during infer
			one_intr=0;
			// the PIR interupt gets disabled in the interupt routine
			// but something keeps enabling it.
			// so just change the PIR pin to pull up to prevent pin transitions
			HalGpioInInit(PIR_PIN,HalGpioPullUp);

			ecm35xx_printf("Motion detected\r\n");

			// turn camera mclk back on then wake camera
			HalGpioWrite(CAM_MCLK_ENA,0);
			HalGpioWrite(CAM_XSLEEP,1);

			LocalTaskDelay(750);

			for(images=3; images; images--){
				//ecm35xx_printf("Infer count %d 0x%x %d\r\n",images,HalGpioRead(PIR_PIN),one_intr);
				infer_B(&conf, &loc);

				// comment out LED toggle for measuring ECM3532 current during infer
				IFLEDON HalGpioOutInit(LED_BLUE,0);
				LocalTaskDelay(750);
				IFLEDON HalGpioOutInit(LED_BLUE,1);
			}

			// wait here if there is still PIR motion, gpio5 = 1
			// continue after 1 second or 2 PIR low
			for(uint32_t ui32Dly=5, ui32Cnt=2; ui32Cnt && ui32Dly; ui32Dly--){
				if(HalGpioRead(PIR_PIN) == 0){
					ui32Cnt--;
				} 
				// after inference enable the PIR pin
				HalGpioIntDisable(GPIO_AB_SEL, PIR_PIN);
				HalGpioIntClear(GPIO_AB_SEL, PIR_PIN);
				HalGpioInInit(PIR_PIN,HalGpioPullDown);
				LocalTaskDelay(200);
				//ecm35xx_printf("Motion count %d 0x%x %d\r\n",ui32Cnt,HalGpioRead(PIR_PIN),one_intr);
			}

			HalGpioIntEnable(GPIO_AB_SEL, PIR_PIN);

			ecm35xx_printf("Motion stopped\r\n");
			// clear Pumpkin wake, clear motion detected during infer, enable motion detection
			HalGpioOutInit(PUMPKIN_PIN,0);
		}

		// amem = xPortGetFreeHeapSize();
		// ecm35xx_printf("Heap In Bytes  @ start %d, @ End %d Mem leakage %d\r\n", bmem, amem, (bmem - amem));

   }

}

int main( void ){
    ExecInit () ;

    // these are critical to obtaining lowest stall current
    // pins uart0_cts, uart0_rx, spi0_miso and swdclk must not float
    // pull either high or low. used low here so current goes to ground and not from vdd_io
    // no pull on swdclk so must have physical resistor.
    REG_RTC_UART0_CTRL.V = 0xa;

    // SPI0 used low here since spi flash is powered off
    REG_RTC_SPI0_CTRL.V = 0x20;
    // spi0 hold in soft reset
    REG_SPI_CONFIG2(0).V = 0x80;
    // bit bang mosi and clock to low.
    REG_SPI_GENERAL_DEBUG(0).V = 0x800;

    // setup pull down on shared camera, als, accel interupt
    // pull down to disable any interrupts
    HalGpioPull(CAM_ALS_ACCL_INT, HalGpioPullDown);
    // board_init enabled the input with no pull, but disable it
    REG_GPIO8_INPUT_ENABLE.V &= ~(0x1<<CAM_ALS_ACCL_INT);

    // setup pull down to turn off spi flash and microphone
    HalGpioPull(SPI_FLASH_POWER, HalGpioPullDown);
    HalGpioPull(PDM_MIC_POWER, HalGpioPullDown);

    // setup PIR GPIO5
    HalGpioInInit(PIR_PIN,HalGpioPullDown);
    // setup GPIO27 wake for Pumpkin
    HalGpioOutInit(PUMPKIN_PIN,0);
    IFLEDON HalGpioOutInit(LED_RED,0);
    IFLEDON HalGpioOutInit(LED_BLUE,0);

    xTaskCreate(vTinyeyeTask, "vInferTask", TASK_STACK_SIZE, NULL, tskIDLE_PRIORITY + 3, NULL);
    //xTaskCreate(vLedBlinkTask, "Blink", TASK_STACK_SIZE, NULL, tskIDLE_PRIORITY + 2, NULL);
    
    /* Start the scheduler. */
    // if only vTaskStartScheduler() is in main GPIO4 turns on to low
    vTaskStartScheduler();

    // if only this while(1) is in main GPIO4 remains off
    //while(1);

    return 0;

}
