/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
vcyou may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_OVX_OVX_OPS_DEFINITIONS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_OVX_OVX_OPS_DEFINITIONS_H_

#include "i_graph_transfer_ops_definitions.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// HVX internal supported ops names
enum class SupportedOpType {
  ADD,
  CONV2D,
  CONVOLUTION_RELUE,
  CONVOLUTION_RELUE_POOL,
  FULLCONNECT,
  FullConnectRelu,
  SOFTMAX_F,
  INPUT,
  OUTPUT,
  NOP,
  OP_CONST, /* OP_ is required to avoid compilation error on windows */
  CHECK,
  CLOSE_FLOAT32,
  CLOSE_QINT8,
  CLOSE_Q_QINT8,
  CLOSE_INT32,
  CLOSE_QINT32,
  PPRINT_8,
  PPRINT_32,
  PPRINT_FLOAT,
  PREFREE,
  FLATTEN,
  QUANTIZEDCONV2D_8X8TO32,
  QUANTIZEDCONV2D_8X8TO32_REF,
  QUANTIZEDMATMUL_8X8TO32,
  QUANTIZEDMATMUL_8X8TO32_REF,
  QUANTIZEDOWNANDSHRINKRANGE_32TO8,
  QUANTIZEDOWNANDSHRINKRANGE_32TO8_REF,
  QUANTIZEDRELU_8,
  QUANTIZEDRELU_8_REF,
  QUANTIZEDRELUX_8,
  QUANTIZEDRELUX_8_REF,
  QUANTIZEDMAXPOOL_8,
  QUANTIZEDMAXPOOL_8_REF,
  QUANTIZEDAVGPOOL_8,
  QUANTIZEDAVGPOOL_8_REF,
  QUANTIZEDCONCAT_8,
  QUANTIZEDCONCAT_8_REF,
  QUANTIZEDBIASADD_8P8TO32,
  QUANTIZEDBIASADD_8P8TO32_REF,
  MIN_F,
  MIN_F_REF,
  MAX_F,
  MAX_F_REF,
  QUANTIZE,
  QUANTIZE_REF,
  DEQUANTIZE,
  DEQUANTIZE_REF,
  SUPERNODE_8X8P8TO8,
  SUPERNODE_8X8P8TO8_REF,
  QUANTIZEDFLATTEN,
  CONV2D_F,
  MATMUL_F,
  RELU_F,
  RELUX_F,
  AVGPOOL_F,
  MAXPOOL_F,
  CONCAT_F,
  BIASADD_F,
  LRN_F,
  VARIABLE,
  ASSIGN,
  RESHAPE,
  QUANTIZED_RESHAPE,
  TANH_F,
  SIGMOID_F,
  SUPPORTED_OP_TYPE_COUNT,
};

// OvxOpsDefinitions provides ops definitons supported in ovx library
// TODO(satok): add a functionality to call functions in ovx library
class OvxOpsDefinitions final : public IGraphTransferOpsDefinitions {
 public:
  static const IGraphTransferOpsDefinitions& getInstance();

  int GetTotalOpsCount() const final;
  int GetOpIdFor(const string& op_type) const final;
  GraphTransferInfo::Destination GetTransferDestination() const final;

 private:
  OvxOpsDefinitions() = default;

  TF_DISALLOW_COPY_AND_ASSIGN(OvxOpsDefinitions);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_OVX_OVX_OPS_DEFINITIONS_H
