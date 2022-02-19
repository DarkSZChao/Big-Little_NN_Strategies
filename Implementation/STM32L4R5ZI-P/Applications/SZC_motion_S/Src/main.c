/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2019 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */


#define SYSCLOCK_MHz 24  //Set system clock
#define INPUT_SIZE sizeof(Input_data_s3[0])  // pass the int8 data to input, if it's other type, then sizeof(input_data)/sizeof(input_data[0])
#define OUTPUT_SIZE sizeof(nnom_output_data_S5)


/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "nnom.h"
#include "single_motion_5_forNNoM_S5.h"
#include "NNoM_Apo_STM_MOTION_Detector_s123_150.h"
#include <stdio.h>
int fputc(int ch, FILE *f){ ITM_SendChar(ch); return ch;}


/* Private function  -----------------------------------------------*/
void SystemClock_MHz(uint32_t freq);
static void MX_GPIO_Init(void);
static void MX_TIM5_Init(void);
/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim5;




int main(void){
	/* MCU Configuration--------------------------------------------------------*/
  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();
  /* Configure the system clock */
  SystemClock_MHz(SYSCLOCK_MHz); // Needs to adjust Prescaler to clock/1000 to show ms for printf
  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM5_Init();


	/* Program starts */
	HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_7);
	HAL_Delay(1000);
	HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_7);
	printf(" Datasets: %s\n\n", DATASETS_NAME);
	printf("=== Inferring ===\n\n");
	uint32_t start_timer_val = __HAL_TIM_GET_COUNTER(&htim5);
	
  /* Initialize model */
	nnom_model_t* model;	
	model = nnom_model_create_S5();
	int accurateCount = 0;
	
	for(int current_sample = 0; current_sample < SAMPLE_COUNT; current_sample++){
		HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_14);

		
		/* Set input data */
		memcpy(nnom_input_data_S5, Input_data_s3[current_sample], INPUT_SIZE); 
		/* Invoke the model */
		model_run(model);

		/* Print all the results */
		printf("\nModel estimate = ");
		for (int i = 0; i < OUTPUT_SIZE; i++){
			printf("[%d] ", nnom_output_data_S5[i]);
		}
		printf("\n");
	
		/* Find the result label with max probability */
		int maxVal = -128;
		int maxIndex = 0;
		for (int i = 0; i < OUTPUT_SIZE; i++){
				if (nnom_output_data_S5[i] > maxVal) {
						maxVal = nnom_output_data_S5[i];
						maxIndex = i;
				}
		}
		printf("Output Label: %d\n", maxIndex);
		printf("Expect Label: %d\n", Output_expect[current_sample]);

		/* Print the accuracy */
		if (maxIndex == Output_expect[current_sample]){
				accurateCount++;
		}
		
		
		HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_14);
		HAL_Delay(50);
	}
	/* Print model details*/
	//model_stat(model);
	model_delete(model);
	uint32_t stop_timer_val = __HAL_TIM_GET_COUNTER(&htim5);
	printf("\nTest set accuracy = %d %%\n", (accurateCount*100)/SAMPLE_COUNT);
	printf("\n Execution Time = %dms\n", stop_timer_val - start_timer_val);
	

	HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_7);
	HAL_Delay(1000);
	HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_7);

  while (1)
  {
	 	HAL_SuspendTick();
		HAL_PWR_EnterSLEEPMode(PWR_LOWPOWERREGULATOR_ON, PWR_SLEEPENTRY_WFE);
		HAL_ResumeTick();
  }
}

void SystemClock_MHz(uint32_t freq)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};

  __HAL_FLASH_SET_LATENCY(FLASH_LATENCY_3);
  HAL_RCC_DeInit();
  
  /* MSI is enabled after System reset, activate PLL with MSI as source */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.MSICalibrationValue = RCC_MSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = freq/2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLP = 7;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if(HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    /* Initialization Error */
    Error_Handler();
  }
  
  /* Select PLL as system clock source and configure the HCLK, PCLK1 and PCLK2 
     clocks dividers */
  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;  
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;  
  if(HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    /* Initialization Error */
    Error_Handler();
  }
}

static void MX_TIM5_Init(void)
{

  /* USER CODE BEGIN TIM5_Init 0 */

  /* USER CODE END TIM5_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM5_Init 1 */

  /* USER CODE END TIM5_Init 1 */
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = SYSCLOCK_MHz*1000;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 0xffffffff;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim5) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim5, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim5, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM5_Init 2 */
	HAL_TIM_Base_Start(&htim5);
	__HAL_TIM_SetCounter(&htim5,0);
  /* USER CODE END TIM5_Init 2 */

}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14|GPIO_PIN_7, GPIO_PIN_RESET);

  /*Configure GPIO pins : PB14 PB7 */
  GPIO_InitStruct.Pin = GPIO_PIN_14|GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

}

uint32_t HAL_Get_time_us(void)
{
	return TIM5->CNT;
}

void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */

  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(char *file, uint32_t line)
{ 
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
