/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef GEMM_WRAPPER_H
#define GEMM_WRAPPER_H

#include <stdbool.h>
#include <stdlib.h>

#include "ovx_nn.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// General functions
void ovx_controller_PrintGraph(uint32_t nn_id);

int ovx_controller_GetWrapperVersion();

int ovx_controller_GetOvxBinaryVersion();

int ovx_controller_DeInitOvx();

int ovx_controller_InitOvx();

uint32_t ovx_controller_GetTargetGraphId();

void ovx_controller_SetTargetGraphId(uint32_t graph_id);

// Graph data transfer functions
int ovx_controller_GetInputNodeData(
        const char* node_name, uint8_t** buf, uint8_t* bytes);

float* ovx_controller_GetOutputNodeData(
    const char* node_name, uint8_t** buf, uint8_t* bytes);

// Graph functions
uint32_t ovx_controller_InstantiateGraph();

void ovx_controller_InitGraph(int version, uint32_t nn_id);

bool ovx_controller_ConstructGraph(uint32_t nn_id);

uint32_t ovx_controller_SetupGraph(int version);

bool ovx_controller_ExecuteGraph(
    const uint32_t nn_id, const uint32_t batches, const uint32_t height,
    const uint32_t width, const uint32_t depth, uint8_t* int_data,
    const uint32_t int_data_size, uint32_t* out_batches, uint32_t* out_height,
    uint32_t* out_width, uint32_t* out_depth, uint8_t* out_vals,
    const uint32_t output_val_byte_size, uint32_t* out_data_byte_size);

void ovx_controller_DumpNodeName(uint32_t nn_id);

int ovx_controller_AppendNode(const char* const name, int graph_id,
                                  int node_id, int op_id, int padding_id,
                                  const ovx_nn_input* const inputs,
                                  int inputs_count,
                                  const ovx_nn_output* const outputs,
                                  int outputs_count);

int ovx_controller_AppendConstTensor(const char* const name, int graph_id,
                                       int node_id, int batch, int height,
                                       int width, int depth,
                                       const uint8_t* const data,
                                       int data_length);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // GEMM_WRAPPER_H
