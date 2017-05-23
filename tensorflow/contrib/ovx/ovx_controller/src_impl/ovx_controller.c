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

#include <stdio.h>

#include "ovx_controller.h"
#include "soc_interface.h"
#include "ovx_log.h"

// if true, show id for each node
static const bool DBG_SHOW_ID = false;

static const uint32_t OUTPUT_PARAM_MAX_LINE_SIZE = 1000;

#define OVX_CONTROLLER_VERSION 0

// allocate print bufsize in advance @MB
#define PRINT_BUFSIZE (2 * 1024 * 1024)

static unsigned char s_print_buf[PRINT_BUFSIZE];

static vsi_nn_graph_t * s_graph;

bool ovx_controller_ExecuteGraph(uint32_t nn_id) {
  bool success;
  if (!success) {
    OVXLOGE("Execution failed");
    return false;
  }

  return true;
}

uint32_t ovx_controller_GetTargetGraphId() {
  return 1;
}

void ovx_controller_SetTargetGraphId(uint32_t graph_id) {
    //TODO
}

void ovx_controller_PrintGraph(uint32_t id) {
}

int ovx_controller_GetWrapperVersion() {
  return OVX_CONTROLLER_VERSION;
}

int ovx_controller_GetOvxBinaryVersion() {
  int retval = 0;
  return retval;
}

int ovx_controller_InitOvx() {
    return 0;
}

int ovx_controller_DeInitOvx() {
  OVXLOGI("Finalize ovx");
  return retval;
}

// Append const tensor to the graph
int ovx_controller_AppendConstTensor(
    const char* const name, int graph_id, int node_id,
    int batch, int height, int width, int depth,
    const uint8_t* const data, int data_length) {
#if 0
  if (DBG_SHOW_ID) {
    OVXLOGV("---(CONST) %s, %d, %d, %d, %d, %d, %d",
            name, node_id, batch, height, width, depth, data_length);
  } else {
    OVXLOGV("---(CONST) %s, %d, %d, %d, %d, %d",
            name, batch, height, width, depth, data_length);
  }
  const int retval = ovx_nn_append_const_node(
      graph_id, node_id, batch, height, width, depth, data, data_length);
  if (retval != 0) {
    OVXLOGE("Failed to append const node %d", node_id);
    return retval;
  }
#endif
  return retval;
}

// Append node to the graph
int ovx_controller_AppendNode(
    const char* const name, int graph_id, int node_id, int ops_id,
    int padding_id, const ovx_nn_input* const inputs,
    int inputs_count, const ovx_nn_output* const outputs,
    int outputs_count) {

  int ovxnode_id = 0;
#if 0
  ovxnode_id = ovx_nn_append_node(
      graph_id, node_id, ops_id, padding_id,
      inputs, inputs_count,
      outputs, outputs_count);

  if (retval != 0) {
    OVXLOGE("Failed to append const node %d", node_id);
    return retval;
#endif
  return ovxnode_id;
}

