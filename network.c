#include "network.h"


NETWORK* createNetwork(int nbInput, int nbOutput)
{
  NETWORK *net = (NETWORK*)malloc(sizeof(NETWORK));
  
  net->nbInput = nbInput;
  //net->nbOutput = nbOutput;
  net->isCompiled = 'n';
  
  net->layers = NULL;
  net->nbLayers = 0;
  
  net->synapses = NULL;
  net->nbSynapses = 0;
  
  net->errors = NULL;
  net->layerTi_1 = NULL;
  net->synapsesTemp = NULL;
  net->synapsesT = NULL;
  
  return net;
}

void compileNetwork(NETWORK *net)
{
  /*if(net->nbLayers == 0)
  {
    MAT *syn = createMatrix_float(net->nbInput, net->nbOutput);
    fillRandomValuesFloat(syn, SYNAPSES_INIT_MIN, SYNAPSES_INIT_MAX);
    addSynapsesToNetwork(net, syn);
    return;
  }*/
  
  for(int i = 0; i < net->nbLayers; i++)
  {
    if(i == 0)
    {
      //MAT *syn = createSynapsesBetweenLayers(net->inputLayer, net->layers[0]);
      MAT *syn = createMatrix_float(net->nbInput, net->layers[0]->matN);
      fillRandomValuesFloat(syn, SYNAPSES_INIT_MIN, SYNAPSES_INIT_MAX);
      addSynapsesToNetwork(net, syn);
    } 
    else 
    {
      MAT *syn = createSynapsesBetweenLayers(net->layers[i-1], net->layers[i]);
      fillRandomValuesFloat(syn, SYNAPSES_INIT_MIN, SYNAPSES_INIT_MAX);
      addSynapsesToNetwork(net, syn);
    }
  }
  
  /*if(net->nbLayers != 0)//Create syn between last and output layers
  {
    //MAT *syn = createSynapsesBetweenLayers(net->layers[net->nbLayers-1], net->outputLayer);
    MAT *syn = createMatrix_float(net->layers[net->nbLayers-1]->matN, net->nbOutput);  
    fillRandomValuesFloat(syn, SYNAPSES_INIT_MIN, SYNAPSES_INIT_MAX);
    addSynapsesToNetwork(net, syn);
  }*/
  
  net->isCompiled = 'y';
}

MAT *createSynapsesBetweenLayers(MAT *l1, MAT *l2)
{
  return createMatrix_float(l1->matN, l2->matN);
}

void addSynapsesToNetwork(NETWORK *net, MAT *syn)
{
  MAT **newSyn = (MAT**)realloc(net->synapses, sizeof(MAT*)*net->nbSynapses + 1);
  if(newSyn == NULL)
  {
    printf("Error while adding synapses\n");
    return;
  }
  net->synapses = newSyn;
  
  net->synapses[net->nbSynapses] = syn;
  net->nbSynapses += 1;
}

void addLayer(NETWORK *net, int nbNeuron)
{
  MAT** newMem = realloc(net->layers, sizeof(MAT*)*net->nbLayers + 1);
  if(newMem == NULL)
  {
    printf("Error to adding layer\n");
    return;
  }
  net->layers = newMem;
  
  net->layers[net->nbLayers] = createMatrix_float(1, nbNeuron);
  net->nbLayers += 1;
}

void printNetworkInfo(NETWORK *net)
{
  printf("Net is compiled: %c\n", net->isCompiled);
  printf("Net input size: %d\n", net->nbInput);
  
  for(int i = 0; i < net->nbLayers; i++)
  {
    printf("Layers %d size: %d\n", i, net->layers[i]->matN);
  }
  //printf("Net output size: %d\n", net->nbOutput);
  
  printf("NbLayers(hidden): %d\nNbSynapses: %d\n\n", net->nbLayers, net->nbSynapses);
}

/*
  Make prediction of output based on the input
  Parameter: 
    - net <->: the neural network
    - input ->: the input matrix, must be matrix of size (1 X inputSize)
    - output <- : the output matrix, must be size (1 X outputSize)
*/
void predict(NETWORK *net, MAT *input, MAT *output)
{
  if(net->nbLayers == 0)
  {
    output = input;
    printf("Warning: no hidden layer, input has been copied to output\n");
  }
  else
  {
    for(int i = 0; i < net->nbLayers; i++)
    {

      if(i == 0)
      {
        //printMatrix(input, "input");
        //printMatrix(net->synapses[0], "syn0");
        matrixDotProduct(input, net->synapses[0], net->layers[0]); //First value
      }
      else
      {
        //printMatrix(net->layers[i-1], "layers i-1");
        //printMatrix(net->synapses[i], "synapses i");
        matrixDotProduct(net->layers[i-1], net->synapses[i], net->layers[i]);
      }
      sigmoid_matrix(net->layers[i], net->layers[i]);
      //printMatrix(net->layers[i], "layers i");
    }

    //printf("End forward propagation\n");
    printMatrix(net->layers[net->nbLayers-1],"End predict");
    if(output != NULL)
    {
      output->mat = net->layers[net->nbLayers - 1]->mat;
      output->matM = net->layers[net->nbLayers - 1]->matM;
      output->matN = net->layers[net->nbLayers - 1]->matN;
    }   
  }
}


void initTraining(NETWORK *net)
{
  //Creating errors
  MAT **newErrors = (MAT**)realloc(net->errors, sizeof(MAT*)*net->nbLayers);
  if(newErrors == NULL)
  {
    printf("Error while creating errors matrix\n");
    return;
  }
  net->errors = newErrors;
  
  //Creating layerTi_1
  MAT **newLayerTi_1 = (MAT**)realloc(net->layerTi_1, sizeof(MAT*)*net->nbLayers);
  if(newLayerTi_1 == NULL)
  {
    printf("Error while creating layerTi_1 matrix\n");
    return;
  }
  net->layerTi_1 = newLayerTi_1;
  
  //Creating synapsesTemp
  MAT **newSynapsesTemp = (MAT**)realloc(net->synapsesTemp, sizeof(MAT*)*net->nbLayers);
  if(newSynapsesTemp == NULL)
  {
    printf("Error while creating synapsesTemp matrix\n");
    return;
  }
  net->synapsesTemp = newSynapsesTemp;
  
  //Creating synapsesT
  MAT **newSynapsesT = (MAT**)realloc(net->synapsesT, sizeof(MAT*)*net->nbLayers);
  if(newSynapsesT == NULL)
  {
    printf("Error while creating synapsesT matrix\n");
    return;
  }
  net->synapsesT = newSynapsesT;
  
  //Create matrices in arrays
  for(int i = 0; i < net->nbLayers; i++)
  {
    net->errors[i] = createMatrix_float(net->layers[i]->matM, net->layers[i]->matN);
    if(i == 0)
    {
      net->layerTi_1[i] = createMatrix_float(net->nbInput, 1);
    }
    else
    {
      net->layerTi_1[i] = createMatrix_float(net->layers[i-1]->matN, net->layers[i-1]->matM);
    }
    net->synapsesTemp[i] = createMatrix_float(net->synapses[i]->matM, net->synapses[i]->matN);
    net->synapsesT[i] = createMatrix_float(net->synapses[i]->matN, net->synapses[i]->matM);
  }
}

void deinitTraining(NETWORK *net)
{
  printf("TODO: free array of MAT\n");
  
}

/*
  Train the network one time based on the dataIn input and dataOut desired output
  Parameters:
    - net <-> : the neural network to be trained
    - dataIn -> : the input data that will be used to train, size
*/
void train(NETWORK *net, MAT *dataIn, MAT *dataOut)
{
  //Init the last error
  //net->layers[net->nbLayers - 1]->mat;
  predict(net, dataIn, NULL);

  matrixSubstract(dataOut, net->layers[net->nbLayers-1], net->errors[net->nbLayers-1]);
  printMatrix(net->errors[net->nbLayers-1], "delta");
  
  for(int i = net->nbLayers - 1; i >= 0; i--)
  {
    printf("Backpropagating layer %d\n", i);
    printMatrix(net->layers[i], "layer i");
    dSigmoid_matrix(net->layers[i], net->layers[i]);
    matrixMultiply(net->errors[i], net->layers[i], net->errors[i]);
    printMatrix(net->errors[i], "error");
    if(i == 0)
    {
      transposeMatrix_float(dataIn, net->layerTi_1[i]);
    } 
    else
    {
      transposeMatrix_float(net->layers[i-1], net->layerTi_1[i]);
    }
    printMatrix(net->layerTi_1[i], "layer t-1");
    matrixDotProduct(net->layerTi_1[i], net->errors[i], net->synapsesTemp[i]);
    printMatrix(net->synapsesTemp[i], "Correction");
    matrixAdd(net->synapses[i], net->synapsesTemp[i], net->synapses[i]);
    if(i != 0){
      transposeMatrix_float(net->synapses[i], net->synapsesT[i]);
      matrixDotProduct(net->errors[i], net->synapsesT[i], net->errors[i - 1]);
    }
  }
}



