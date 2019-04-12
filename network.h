#ifndef NETWORK_H
#define NETWORK_H

#include "ai_tools.h"

#define SYNAPSES_INIT_MIN -5
#define SYNAPSES_INIT_MAX 5

typedef struct {
  int nbInput;
  //int nbOutput;
  char isCompiled; /* 'y' if yes, 'n' if no */
  /*
    Info: Store the outputs of each computation (results of layers)
  */
  MAT **layers;
  int nbLayers;
  /*
    Info: Store the "data" of the network, this is used to link layers to each other
  */
  MAT **synapses;
  int nbSynapses;
  
  MAT **errors; //Matrices to store errors on each layers
  MAT **layerTi_1; //Matrices to store the transpose of layers
  MAT **synapsesTemp; //Matrices to store temp value on synapses
  MAT **synapsesT; //Matrices to store transpose of synapses
  
} NETWORK;

//API functions
NETWORK* createNetwork(int nbInput, int nbOutput);
void compileNetwork(NETWORK* net); //To create synapses between layers
void addLayer(NETWORK* net, int nbNeuron);
void printNetworkInfo(NETWORK *net);
void predict(NETWORK *net, MAT *input, MAT *output);

void initTraining(NETWORK *net);
void deinitTraining(NETWORK *net);
void train(NETWORK *net, MAT *dataIn, MAT *dataOut);

//Internal functions
MAT *createSynapsesBetweenLayers(MAT *l1, MAT *l2);
void addSynapsesToNetwork(NETWORK *net, MAT *syn);

#endif
