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

#include "soc_interface.h"

#include <inttypes.h>

#include "ovx_controller.h"
#include "ovx_nn.h"
#include "node_data_float.h"
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
  ovx_controller_InitOvx();
  return true;
}

bool soc_interface_Finalize() {
  OVXLOGD("Finalize");
  ovx_controller_DeInitOvx();
  return true;
}

bool soc_interface_ExecuteGraph() {
  bool success;
  OVXLOGD("ExecuteGraph");
  const uint32_t graph_id = ovx_controller_GetTargetGraphId();
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
    int x, int y, int z, int d, const uint8_t* const buf,
    uint64_t buf_size) {
  OVXLOGD("FillInputNodeFloat");
  return true;
}

bool soc_interface_ReadOutputNode(
    const char* const node_name, uint8_t** buf, uint64_t *buf_size) {
  OVXLOGD("ReadOutputNodeFloat");
  return true;
}

// Append const node to the graph
bool soc_interface_AppendConstTensor(
    const char* const name, int node_id, int batch, int height, int width, int depth,
    const uint8_t* const data, int data_length) {
  const uint32_t graph_id = ovx_controller_GetTargetGraphId();
  const int retval = ovx_controller_AppendConstTensor(
      name, graph_id, node_id, batch, height, width, depth, data, data_length);
  if (retval != 0) {
    OVXLOGE("Failed to append const tensor %d", node_id);
    return false;
  }
  return true;
}

// Append node to the graph
int soc_interface_AppendNode(
    const char* const name, int node_id, int ops_id, int padding_id, const void* const inputs,
    int inputs_count, const void* const outputs, int outputs_count) {
  const uint32_t graph_id = ovx_controller_GetTargetGraphId();
  const int ovxnode_id = ovx_controller_AppendNode(
      name, graph_id, node_id, ops_id, padding_id,
      (ovx_nn_input*) inputs, inputs_count,
      (ovx_nn_output*) outputs, outputs_count);
  return (int)ovxnode_id;
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
