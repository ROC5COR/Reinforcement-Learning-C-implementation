#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ai_tools.h"
#include "network.h"

uint32_t totalTrainingTime = 0;

MAT *createAndFill(int size, float a, float b)
{
  MAT *out = createMatrix_float(1, size);
  if(size == 1)
  {
    ((VAR**)out->mat)[0][0] = a;  
  }
  else
  {
    ((VAR**)out->mat)[0][0] = a;
    ((VAR**)out->mat)[0][1] = b;
  }
  return out;
}

int main(int argc, char *argv[])
{
  printf("\n\nDesktop RL implementation\n");

  srand(time(NULL));

  
  NETWORK *net = createNetwork(2, 1);
  addLayer(net, 3);//Hidden layer
  addLayer(net, 3);
  addLayer(net, 3);
  addLayer(net, 1);//Output layer
  printNetworkInfo(net);
  compileNetwork(net);
  printNetworkInfo(net);

  MAT *in1 = createMatrix_float(1, 2); 
  ((VAR**)in1->mat)[0][0] = 1.0; ((VAR**)in1->mat)[0][1] = 1.0;
  MAT *in2 = createMatrix_float(1, 2); 
  ((VAR**)in2->mat)[0][0] = 1.0; ((VAR**)in2->mat)[0][1] = 0.0;
  MAT *in3 = createMatrix_float(1, 2); 
  ((VAR**)in3->mat)[0][0] = 0.0; ((VAR**)in3->mat)[0][1] = 1.0;
  MAT *in4 = createMatrix_float(1, 2); 
  ((VAR**)in4->mat)[0][0] = 0.0; ((VAR**)in4->mat)[0][1] = 0.0;
  //MAT *in2 = createAndFill(2, 1.0, 0.0);
  //MAT *in3 = createAndFill(2, 0.0, 1.0);
  //MAT *in4 = createAndFill(2, 0.0, 0.0);

  MAT *out1 = createMatrix_float(1, 1); ((VAR**)out1->mat)[0][0] = 0.0;
  MAT *out2 = createMatrix_float(1, 1); ((VAR**)out2->mat)[0][0] = 1.0;
  MAT *out3 = createMatrix_float(1, 1); ((VAR**)out3->mat)[0][0] = 1.0;
  MAT *out4 = createMatrix_float(1, 1); ((VAR**)out4->mat)[0][0] = 0.0;
  //MAT *out3 = createAndFill(1, 1.0, 0.0);
  //MAT *out4 = createAndFill(1, 0.0, 0.0);

  printMatrix(in1, "Input");
  MAT *matOut = createMatrix_float(0, 0); 
  predict(net, in1, matOut);
  printMatrix(matOut, "matOutPredicted");
  
  initTraining(net);
  
  for(int i = 0; i < 10; i++)
  {
    train(net, in1, out1);
    train(net, in2, out2);
    train(net, in3, out3);
    train(net, in4, out4);
  }
  printMatrix(in1, "in1");
  printMatrix(out1, "out1");
  predict(net, in1, matOut);
  printMatrix(matOut, "1 matOutPredicted (trained)");
  predict(net, in2, matOut);
  printMatrix(matOut, "2 matOutPredicted (trained)");
  
  deinitTraining(net);
  printf("End\n\n");

  return 0;  

}
