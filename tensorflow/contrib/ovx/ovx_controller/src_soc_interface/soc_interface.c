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

#include <inttypes.h>

#include "soc_interface.h"
#include "ovx_controller.h"
#include "ovx_log.h"

int soc_interface_GetWrapperVersion() {
  OVXLOGD("GetWrapperVersion");
  return ovx_controller_GetWrapperVersion();
}

int soc_interface_GetSocControllerVersion() {
  OVXLOGD("GetSocControllerVersion");
  return ovx_controller_GetOvxBinaryVersion();
}

bool soc_interface_Init() {
  OVXLOGD("Init");
  return ovx_controller_InitOvx();
}

bool soc_interface_Finalize() {
  OVXLOGD("Finalize");
  return ovx_controller_DeInitOvx();
}

bool soc_interface_ExecuteGraph() {
  bool success;
  const uint32_t graph_id = ovx_controller_GetTargetGraphId();

  OVXLOGD("ExecuteGraph");
  if (OVX_CONTROLLER_GRAPH_ID_NA == graph_id) {
    OVXLOGE("Graph id has not been set yet.");
    return false;
  }
  success = ovx_controller_ExecuteGraph(graph_id);
  return success;
}

bool soc_interface_TeardownGraph() {
  OVXLOGD("TeardownGraph");
  return true;
}

bool soc_interface_FillInputNode(
    const char* const name, int node_id,
    const uint32_t * const shape, uint32_t dim_num,
    const uint8_t* const buf, uint64_t buf_size) {
  OVXLOGD("FillInputNode %s", name);
  bool ret = ovx_controller_FillInputNode(
          name, node_id,
          shape, dim_num, buf, buf_size);
  return ret;
}

uint64_t soc_interface_ReadOutputNode(
    const char* const node_name, uint8_t** buf, uint64_t *bytes) {
  OVXLOGD("ReadOutputNode %s", node_name);
  return ovx_controller_GetOutputNodeData(node_name, buf, bytes);
}

// Append const node to the graph
bool soc_interface_AppendConstTensor(
    const char* const name, int node_id,
    const uint32_t * const shape, uint32_t dim_num,
    uint8_t* data, int data_length) {
  bool ret = ovx_controller_AppendConstTensor(
                        name, node_id,
                        shape, dim_num, data, data_length);
  if (true != ret) {
    OVXLOGE("Failed to append const tensor %s", name);
  }
  return ret;
}

// Append node to the graph
uint32_t soc_interface_AppendNode(
    const char* const name, int node_id, int op_id,
    const void* const inputs, int inputs_count,
    const void* const outputs, int outputs_count) {
  const uint32_t graph_id = ovx_controller_GetTargetGraphId();
  const uint32_t ovxnode_id = ovx_controller_AppendNode(
                          name, graph_id, node_id, op_id,
                          inputs, inputs_count,
                          outputs, outputs_count);
  return ovxnode_id;
}


// Instantiate graph
bool soc_interface_InstantiateGraph() {
  const uint32_t nn_id = ovx_controller_InstantiateGraph();
  ovx_controller_SetTargetGraphId(nn_id);
  return true;
}

// Construct graph
bool soc_interface_ConstructGraph() {
  const uint32_t graph_id = ovx_controller_GetTargetGraphId();
  return ovx_controller_ConstructGraph(graph_id);
}

void soc_interface_SetLogLevel(int log_level) {
  SetLogLevel(log_level);
}

void soc_interface_SetDebugFlag(uint64_t flag) {
  OVXLOGI("Set debug flag 0x%" PRIx64, flag);
}

