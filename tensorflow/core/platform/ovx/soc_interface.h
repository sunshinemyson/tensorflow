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

#ifndef TENSORFLOW_PLATFORM_OVX_SOC_INTERFACE_H_
#define TENSORFLOW_PLATFORM_OVX_SOC_INTERFACE_H_

#include <inttypes.h>

// Declaration of APIs provided by ovx shared library. This header is shared
// with both ovx library built with qualcomm SDK and tensorflow.
// All functions defined here must have prefix "soc_interface" to avoid
// naming conflicts.
#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif  // __cplusplus
// Returns the version of loaded ovx wrapper shared library.
// You should assert that the version matches the expected version before
// calling APIs defined in this header.
int soc_interface_GetWrapperVersion();
// Returns the version of ovx binary.
// You should assert that the version matches the expected version before
// calling APIs defined in this header.
int soc_interface_GetSocControllerVersion();
// Initialize SOC
bool soc_interface_Init();
// Finalize SOC
bool soc_interface_Finalize();
// Execute graph on SOC
bool soc_interface_ExecuteGraph();
// Teardown graph setup
bool soc_interface_TeardownGraph();
// Send input data to SOC
bool soc_interface_FillInputNode(const char* const name, int node_id,
                                   const uint32_t * const shape, uint32_t dim_num,
                                   const uint8_t* const buf, uint64_t buf_size);
// Load output data from SOC
uint64_t soc_interface_ReadOutputNode(const char* const node_name,
                                       uint8_t** buf, uint64_t* bytes);

// Append const node to the graph
bool soc_interface_AppendConstTensor(const char* const name, int node_id,
                                    const uint32_t * const shape, uint32_t dim_num,
                                    uint8_t* data, int data_length);

// Append node to the graph
uint32_t soc_interface_AppendNode(const char* const name, int node_id, int op_id,
                                    const void* const inputs, int inputs_count,
                                    const void* const outputs, int outputs_count);

// Instantiate graph
bool soc_interface_InstantiateGraph();

// Construct graph
bool soc_interface_ConstructGraph();

// Set log level
void soc_interface_SetLogLevel(int log_level);

// Set debug flag
void soc_interface_SetDebugFlag(uint64_t flag);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_PLATFORM_OVX_SOC_INTERFACE_H_
