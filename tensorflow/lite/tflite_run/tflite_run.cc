//
// This is a minimal program to perform interpreted inference of
//  a tflite model with given input and check the result against
//  a given output golden. The input and output tensor can be 
//  generated using Acuity toolkit. This program is only supplied
//  as an example.
// 
// TFlite_run will load libneuralnetwork.so in LD_LIBRARY_PATH 
//  and use the NNAPI delegate as backend if the library can
//  be located. Otherwise, it will use the built-in CPU
//  implementation based on gemmlowp/ 
// 
// Usage: tflite_run <tflite model> <input tensor> <output golden>
//

#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

void getTop_u(uint8_t *pfProb, int outputCount, int topNum)
{
    int i,j;
    int MaxClass[5];
    uint8_t fMaxProb[5];

    memset(fMaxProb, 0, sizeof(uint8_t) * topNum);
    memset(MaxClass, 0, sizeof(int) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i=0; i<outputCount; i++)
        {
            if ((i == *(MaxClass+0)) || (i == *(MaxClass+1)) || (i == *(MaxClass+2)) ||
                    (i == *(MaxClass+3)) || (i == *(MaxClass+4)))
                continue;

            if (pfProb[i] > *(fMaxProb+j))
            {
                *(fMaxProb+j) = pfProb[i];
                *(MaxClass+j) = i;
            }
        }
    }
    printf(" --- Top5 ---\n");
    for (i=0; i<5; i++)
    {
        printf("%3d: %d\n", MaxClass[i], fMaxProb[i]);
    }
}

void getTop_f(float *pfProb, int outputCount, int topNum)
{
    int i,j;
    int MaxClass[5];
    float fMaxProb[5];

    memset(fMaxProb, 0, sizeof(float) * topNum);
    memset(MaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i=0; i<outputCount; i++)
        {
            if ((i == *(MaxClass+0)) || (i == *(MaxClass+1)) || (i == *(MaxClass+2)) ||
                    (i == *(MaxClass+3)) || (i == *(MaxClass+4)))
                continue;

            if (pfProb[i] > *(fMaxProb+j))
            {
                *(fMaxProb+j) = pfProb[i];
                *(MaxClass+j) = i;
            }
        }
    }
    printf(" --- Top5 ---\n");
    for (i=0; i<5; i++)
    {
        printf("%3d: %8.6f\n", MaxClass[i], (float)fMaxProb[i]);
    }
}

int main(int argc, char* argv[]) {
  if(argc != 4) {
    fprintf(stderr, "tflite_run <tflite model> <input tensor> <output golden>\n");
    return 1;
  }
  const char* filename = argv[1];
  const char* input_tensor = argv[2];
  const char* output_golden = argv[3];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  printf( "Loaded model %s \n" , filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model.get(), resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  interpreter->UseNNAPI(1);

  int input = interpreter->inputs()[0];
  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  tflite::PrintInterpreterState(interpreter.get());

  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  printf("input dim  %d %d %d \n",dims->data[1],dims->data[2],dims->data[3]);
  uint8_t *uint8_ptr=NULL;
  float *float_ptr=NULL;
  char str[256];
  FILE *fp=fopen(input_tensor,"rb");

  switch (interpreter->tensor(input)->type) {
    case kTfLiteFloat32:
      float_ptr = interpreter->typed_tensor<float>(input);
      while(!feof(fp))
      {
         memset(str,0,sizeof(str));
         fgets(str,sizeof(str),fp);
         *float_ptr = atof(str);
         float_ptr ++;
      }
      break;
    case kTfLiteUInt8:
      uint8_ptr = interpreter->typed_tensor<uint8_t>(input);
      while(!feof(fp))
      {
         memset(str,0,sizeof(str));
         fgets(str,sizeof(str),fp);
         *uint8_ptr = atoi(str);
         uint8_ptr ++;
      }
      break;
    default:
      printf( "cannot handle input type %d \n", interpreter->tensor(input)->type );
      exit(-1);
  }
  fclose(fp);

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  tflite::PrintInterpreterState(interpreter.get());

  int output = interpreter->outputs()[0];
  dims = interpreter->tensor(output)->dims;
  printf("output dim  %d %d %d \n",dims->data[1],dims->data[2],dims->data[3]);
  const int output_size = dims->data[1];
  int error_item=0;
  float golden_f=0.0;
  uint8_t golden_u=0.0;
  fp = fopen(output_golden,"rb");
  memset(str,0,sizeof(str));
  sprintf(str,"output_%d.txt",output_size);
  FILE *out=fopen(str,"wb");
  switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
      float_ptr = interpreter->typed_output_tensor<float>(0);
      while(!feof(fp))
      {
         memset(str,0,sizeof(str));
         fgets(str,sizeof(str),fp);
         golden_f=atof(str);
         if(golden_f < 0.001 || *float_ptr < 0.001)
         {
         if(fabs(*float_ptr - golden_f)>0.0001)
         {
             printf("error %f vs %f\n",*float_ptr,golden_f);
             error_item++;
         }
         }
         else if( fabs(*float_ptr - golden_f)/fabs(golden_f)>0.01)
         {
            printf("error %f vs %f, error_ratio %f\n",*float_ptr,golden_f, fabs(*float_ptr - golden_f)/fabs(golden_f));
            error_item++;
         }
         fprintf(out,"%f\n",*float_ptr);
         float_ptr ++;
      }
      float_ptr = interpreter->typed_output_tensor<float>(0);
      getTop_f(float_ptr,output_size,5);
      break;

    case kTfLiteUInt8:
      uint8_ptr = interpreter->typed_output_tensor<uint8_t>(0);
      while(!feof(fp))
      {
         memset(str,0,sizeof(str));
         fgets(str,sizeof(str),fp);
         golden_u=atoi(str);
         if(golden_u != 0 && fabs(*uint8_ptr - golden_u)/(float)golden_u>0.01)
         {
            printf("error %d vs %d\n",*uint8_ptr,golden_u);
            error_item++;
         }
         fprintf(out,"%d\n",*uint8_ptr);
         uint8_ptr ++;
      }
      uint8_ptr = interpreter->typed_output_tensor<uint8_t>(0);
      getTop_u(uint8_ptr,output_size,5);
      break;

    default:
      printf( "cannot handle output type %d \n", interpreter->tensor(input)->type );
      exit(-1);
  }
  fclose(fp);
  fclose(out);

  if(error_item)
  {
     printf("result mismatched\nfail\n");
  }
  else
  {
     printf("result matched \npass\n");
  }

  return 0;
}
