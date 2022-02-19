/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/common.h"
#include "./model/model.h"
#include "./model/SparkFun_MOTION_Detector.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


// Helper fn to log the shape and datatype of a tensor
void printTensorDetails(TfLiteTensor* tensor, tflite::ErrorReporter* error_reporter) {
  error_reporter->Report("Type [%s] Shape :", TfLiteTypeGetName(tensor->type));
  for (int d = 0; d < tensor->dims->size; ++d) {
    error_reporter->Report("%d [%d]", d, tensor->dims->data[d]);
  }
  error_reporter->Report("");
}


int main(int argc, char* argv[]) {
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.\n", model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }
  // This pulls in all the operation implementations we need.
  tflite::ops::micro::AllOpsResolver resolver;
  // Create an area of memory to use for input, output, & intermediate arrays.
  // The size of this will depend on the model you're using, currently
  // determined by experimentation.
  const int tensor_arena_size = 40 * 1024;
  uint8_t tensor_arena[tensor_arena_size];
  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);
  interpreter.AllocateTensors();


  error_reporter->Report("Parsing Motion classifier model FlatBuffer called [%s], size [%d] bytes.", model_tflite_name, model_tflite_len);
  error_reporter->Report("The name of datasets is [%s]", DATASETS_NAME);
  error_reporter->Report("Arena allocated: %d*1024", tensor_arena_size / 1024);
  // Get information about the models input and output tensors.
  TfLiteTensor* model_input = interpreter.input(0);
  error_reporter->Report("Details of input tensor:");
  printTensorDetails(model_input, error_reporter);
  TfLiteTensor* model_output = interpreter.output(0);
  error_reporter->Report("Details of output tensor:");
  printTensorDetails(model_output, error_reporter);


  // perform inference on each test sample and evalute accuracy of model
  int accurateCount = 0;
  const int inputTensorSize = 128 * 3; 
  error_reporter->Report("Total number of samples is %d\n", SAMPLE_COUNT);

  for (int s = 0; s < SAMPLE_COUNT; s++) {
    // Set value of input tensor
    for (int d = 0; d < inputTensorSize; d++) {
      model_input->data.uint8[d] = Input[s][d];
    }

    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed.\n");
      return 1; 
    }
    
    // find the index of the maximum output value
    int maxima = 0;
    int index = 0;
    for (int i = 0; i < 2; i++) {
      if (maxima < model_output->data.uint8[i]) {
    	maxima = model_output->data.uint8[i];
    	index = i;
      }
      else {
        maxima = maxima;
    	index = index;
      }
    }
    if (index == Output[s]) {
      ++accurateCount;
    }
    
    error_reporter->Report("Model estimate [%d] [%d]", model_output->data.uint8[0], model_output->data.uint8[1]);
    error_reporter->Report("Output label [%d]",index);
    error_reporter->Report("Expect label [%d]\n",Output[s]);
  }

  error_reporter->Report("Activity detector completed successfully.");
  error_reporter->Report("Test set accuracy is [%d] percent\n", (accurateCount * 100) / SAMPLE_COUNT);
}
