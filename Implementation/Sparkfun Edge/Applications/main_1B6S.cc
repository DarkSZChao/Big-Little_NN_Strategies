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
#include "tensorflow/lite/micro/examples/szc_merged_motions/model/model.h"
#include "tensorflow/lite/micro/examples/szc_merged_motions/model/SparkFun_MOTION_Detector.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


int accurateCount = 0;
int index_B = 0;
bool BIG_model_activate = true;


// Helper fn to log the shape and datatype of a tensor
void print_TensorDetails(TfLiteTensor* tensor, tflite::ErrorReporter* error_reporter) {
    error_reporter->Report("Type [%s] Shape :", TfLiteTypeGetName(tensor->type));
    for (int d = 0; d < tensor->dims->size; ++d) {
        error_reporter->Report("%d [%d]", d, tensor->dims->data[d]);
    }
    error_reporter->Report("");
}


void print_IO_type(const int tensor_arena_size){
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    tflite::ops::micro::AllOpsResolver resolver;
    uint8_t tensor_arena[tensor_arena_size];

    tflite::Model* model_S = ::tflite::GetModel(single_motion0_model_tflite);
    tflite::MicroInterpreter interpreter_S(model_S, resolver, tensor_arena, tensor_arena_size, error_reporter);
    interpreter_S.AllocateTensors();
    TfLiteTensor* model_input_S = interpreter_S.input(0);
    TfLiteTensor* model_output_S = interpreter_S.output(0);
    error_reporter->Report("Input tensor of SMALL and BIG:");
    print_TensorDetails(model_input_S, error_reporter);
    error_reporter->Report("Output tensor of SMALL:");
    print_TensorDetails(model_output_S, error_reporter);

    tflite::Model* model_B = ::tflite::GetModel(multi_motions_model_tflite);
    tflite::MicroInterpreter interpreter_B(model_B, resolver, tensor_arena, tensor_arena_size, error_reporter);
    interpreter_B.AllocateTensors();
    TfLiteTensor* model_output_B = interpreter_B.output(0);
    error_reporter->Report("Output tensor of BIG:");
    print_TensorDetails(model_output_B, error_reporter);
}


void Infer_SMALL(tflite::Model* model, int current_sample, const int tensor_arena_size){
    // Initialize the components for model interpreter
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    tflite::ops::micro::AllOpsResolver resolver;
    uint8_t tensor_arena[tensor_arena_size];

    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);
    interpreter.AllocateTensors();
    TfLiteTensor* model_input = interpreter.input(0);
    TfLiteTensor* model_output = interpreter.output(0);

    const int input_tensor_size = 128 * 3;
    // Set value of input tensor
    for (int d = 0; d < input_tensor_size; d++) {
        model_input->data.uint8[d] = Input[current_sample][d];
    }
    // Invoke the model
    interpreter.Invoke();

    // Find the index of the maximum output value
    int maxima = -128;
    int index_S = 0;
    for (int i = 0; i < 2; i++) {
        if (maxima < model_output->data.uint8[i]) {
            maxima = model_output->data.uint8[i];
            index_S = i;
        }
        else {
            maxima = maxima;
            index_S = index_S;
        }
    }
    error_reporter->Report(" Model estimate [%d] [%d]", model_output->data.uint8[0], model_output->data.uint8[1]);
    switch (index_S){
        case 0:
            BIG_model_activate = true;
            error_reporter->Report(" Output label changes, invoke BIG");
            break;
        case 1:
            BIG_model_activate = false;
            error_reporter->Report(" Output label remains: [%d]",index_B);
            if (index_B == Output[current_sample]){
                accurateCount++;
            }
            break;
    }
    error_reporter->Report(" Expect label [%d]\n",Output[current_sample]);
}


void Infer_BIG(tflite::Model* model, int current_sample, const int tensor_arena_size){
    // Initialize the components for model interpreter
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    tflite::ops::micro::AllOpsResolver resolver;
    uint8_t tensor_arena[tensor_arena_size];

    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);
    interpreter.AllocateTensors();
    TfLiteTensor* model_input = interpreter.input(0);
    TfLiteTensor* model_output = interpreter.output(0);

    const int input_tensor_size = 128 * 3;
    // Set value of input tensor
    for (int d = 0; d < input_tensor_size; d++) {
        model_input->data.uint8[d] = Input[current_sample][d];
    }
    // Invoke the model
    interpreter.Invoke();

    // Find the index of the maximum output value
    int maxima = -128;
    index_B = 0;
    for (int i = 0; i < 6; i++) {
        if (maxima < model_output->data.uint8[i]) {
            maxima = model_output->data.uint8[i];
            index_B = i;
        }
        else {
            maxima = maxima;
            index_B = index_B;
        }
    }
    if (index_B == Output[current_sample]){
        accurateCount++;
    }
    error_reporter->Report(" Model estimate [%d] [%d] [%d] [%d] [%d] [%d]", model_output->data.uint8[0], model_output->data.uint8[1], model_output->data.uint8[2],
                                                                            model_output->data.uint8[3], model_output->data.uint8[4], model_output->data.uint8[5]);
    error_reporter->Report(" Output label [%d]",index_B);
    error_reporter->Report(" Expect label [%d]\n",Output[current_sample]);
    BIG_model_activate = false;
}


int main(int argc, char* argv[]) {
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    const int tensor_arena_size = 50 * 1024;
    tflite::Model* model;

    error_reporter->Report("\n\n---------------------------------------------------------------------------------------------------------------------");
    error_reporter->Report("Parsing Motion classifier model FlatBuffer: [%s], size [%d] bytes.", single_motion0_model_tflite_name, single_motion0_model_tflite_len);
    error_reporter->Report("                                            [%s], size [%d] bytes.", single_motion1_model_tflite_name, single_motion1_model_tflite_len);
    error_reporter->Report("                                            [%s], size [%d] bytes.", single_motion2_model_tflite_name, single_motion2_model_tflite_len);
    error_reporter->Report("                                            [%s], size [%d] bytes.", single_motion3_model_tflite_name, single_motion3_model_tflite_len);
    error_reporter->Report("                                            [%s], size [%d] bytes.", single_motion4_model_tflite_name, single_motion4_model_tflite_len);
    error_reporter->Report("                                            [%s], size [%d] bytes.", single_motion5_model_tflite_name, single_motion5_model_tflite_len);
    error_reporter->Report("                                            [%s], size [%d] bytes.", multi_motions_model_tflite_name, multi_motions_model_tflite_len);
    error_reporter->Report("Arena allocated: %d*1024", tensor_arena_size / 1024);
    error_reporter->Report("The name of datasets is [%s]", DATASETS_NAME);
    error_reporter->Report("Total number of samples is %d\n", SAMPLE_COUNT);
    print_IO_type(tensor_arena_size);
    error_reporter->Report("---------------------------------------------------------------------------------------------------------------------\n\n");


    for (int current_sample = 0; current_sample < SAMPLE_COUNT; current_sample++) {
        if (!BIG_model_activate){
            switch (index_B){
                case 0:
                    error_reporter->Report("[0th] SMALL model is inferring:");
                    model = ::tflite::GetModel(single_motion0_model_tflite);
                    Infer_SMALL(model, current_sample, tensor_arena_size);
                    break;
                case 1:
                    error_reporter->Report("[1th] SMALL model is inferring:");
                    model = ::tflite::GetModel(single_motion1_model_tflite);
                    Infer_SMALL(model, current_sample, tensor_arena_size);
                    break;
                case 2:
                    error_reporter->Report("[2th] SMALL model is inferring:");
                    model = ::tflite::GetModel(single_motion2_model_tflite);
                    Infer_SMALL(model, current_sample, tensor_arena_size);
                    break;
                case 3:
                    error_reporter->Report("[3th] SMALL model is inferring:");
                    model = ::tflite::GetModel(single_motion3_model_tflite);
                    Infer_SMALL(model, current_sample, tensor_arena_size);
                    break;
                case 4:
                    error_reporter->Report("[4th] SMALL model is inferring:");
                    model = ::tflite::GetModel(single_motion4_model_tflite);
                    Infer_SMALL(model, current_sample, tensor_arena_size);
                    break;
                case 5:
                    error_reporter->Report("[5th] SMALL model is inferring:");
                    model = ::tflite::GetModel(single_motion5_model_tflite);
                    Infer_SMALL(model, current_sample, tensor_arena_size);
                    break;
                default:
                    BIG_model_activate = true;
                    break;
            }
        }
        if (BIG_model_activate){
            error_reporter->Report("BIG model is inferring:");
            model = ::tflite::GetModel(multi_motions_model_tflite);
            Infer_BIG(model, current_sample, tensor_arena_size);
        }
    }
    error_reporter->Report("Activity detector completed successfully.");
    error_reporter->Report("Test set accuracy is [%d] percent\n", (accurateCount * 100) / SAMPLE_COUNT);
}
