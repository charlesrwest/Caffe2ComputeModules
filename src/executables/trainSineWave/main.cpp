#include<cstdlib>
#include<string>

#include "caffe2/core/workspace.h"
#include "caffe2/core/tensor.h"
#include<google/protobuf/text_format.h>
#include "FullyConnectedModuleDefinition.hpp"
#include<iostream>
#include<cmath>
#include<cassert>
#include "AveragedL2LossModuleDefinition.hpp"
#include "AdamSolver.hpp"

const double PI  =3.141592653589793238463;
const float  PI_F=3.14159265358979f;

template<class DataType>
void PairedRandomShuffle(typename std::vector<DataType>& inputData, typename  std::vector<DataType>& expectedOutputData)
{
assert(inputData.size() == expectedOutputData.size());

//Fisher-Yates shuffle
for(typename std::vector<DataType>::size_type index = 0; index < inputData.size(); index++)
{
typename std::vector<DataType>::size_type elementToSwapWithIndex = index + (rand() % (inputData.size() - index));
std::swap(inputData[index], inputData[elementToSwapWithIndex]);
std::swap(expectedOutputData[index], expectedOutputData[elementToSwapWithIndex]);
}
};

int main(int argc, char **argv)
{
//Make 10000 training examples
int64_t numberOfTrainingExamples = 10000;
std::vector<float> trainingInputs;
std::vector<float> trainingExpectedOutputs;

for(int64_t trainingExampleIndex = 0; trainingExampleIndex < numberOfTrainingExamples; trainingExampleIndex++)
{
trainingInputs.emplace_back((((double) trainingExampleIndex)/(numberOfTrainingExamples+1))*2.0 - 1.0);
trainingExpectedOutputs.emplace_back(sin(((double) trainingExampleIndex)/(numberOfTrainingExamples+1)*2.0*PI)*.5 + .5);
}

//Shuffle the examples
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);

//Create the Caffe2 workspace/context
caffe2::Workspace workspace;
caffe2::CPUContext context;

//Create the blobs to inject the sine input/output for training
caffe2::TensorCPU& inputBlob = *workspace.CreateBlob("inputBlob")->GetMutable<caffe2::TensorCPU>();
inputBlob.Resize(1, 1);
inputBlob.mutable_data<float>();

std::string trainingExpectedOutputBlobName = "trainingExpectedOutputBlobName";
caffe2::TensorCPU& expectedOutputBlob = *workspace.CreateBlob(trainingExpectedOutputBlobName)->GetMutable<caffe2::TensorCPU>();
expectedOutputBlob.Resize(1, 1);
expectedOutputBlob.mutable_data<float>();

//Define a 3 layer fully connected network with default (sigmoidal) activation
GoodBot::FullyConnectedModuleDefinitionParameters networkParameters;
networkParameters.inputBlobName = "inputBlob";
networkParameters.numberOfInputs = 1;
networkParameters.numberOfNodesInLayers = {100, 100, 1};
networkParameters.moduleName = "HelloNetwork";

GoodBot::FullyConnectedModuleDefinition network(networkParameters);
network.SetMode("TRAIN");

//Add a module for computing loss to the end of the network
GoodBot::AveragedL2LossModuleDefinitionParameters lossParameters;
lossParameters.inputBlobName = network.GetOutputBlobNames()[0];
lossParameters.moduleName = "AveragedL2Loss";
lossParameters.trainingExpectedOutputBlobName = trainingExpectedOutputBlobName;
lossParameters.testExpectedOutputBlobName = trainingExpectedOutputBlobName;

network.AddModule(*(new  GoodBot::AveragedL2LossModuleDefinition(lossParameters)));
network.SetMode("TRAIN");

//Add a solver module for training/updating
GoodBot::AdamSolverParameters solverParams;
solverParams.moduleName = "AdamSolver";
solverParams.trainableParameterNames = network.GetTrainableBlobNames();
solverParams.trainableParameterShapes = network.GetTrainableBlobShapes();

network.AddModule(*(new GoodBot::AdamSolver(solverParams)));

//Add function to allow printing of network architectures
std::function<void(const google::protobuf::Message&)> print = [&](const google::protobuf::Message& inputMessage)
{
std::string buffer;

google::protobuf::TextFormat::PrintToString(inputMessage, &buffer);

std::cout << buffer<<std::endl;
};

//Training the network, so set the mode to train
network.SetMode("TRAIN");

//Initialize the network by automatically generating the NetDef for network initialization in "TRAIN" mode
caffe2::NetDef trainingNetworkInitializationDefinition = network.GetInitializationNetwork();

//Print out the generated network architecture
print(trainingNetworkInitializationDefinition);

//Create and run the initialization network.
caffe2::NetBase* initializationNetwork = workspace.CreateNet(trainingNetworkInitializationDefinition);
initializationNetwork->Run();

//Automatically generate the training network
caffe2::NetDef trainingNetworkDefinition = network.GetNetwork(workspace.Blobs());

print(trainingNetworkDefinition);

//Instance the training network implementation
caffe2::NetBase* trainingNetwork = workspace.CreateNet(trainingNetworkDefinition);


//Create the deploy version of the network
network.SetMode("DEPLOY");

caffe2::NetDef deployNetworkDefinition = network.GetNetwork(workspace.Blobs());

caffe2::NetBase* deployNetwork = workspace.CreateNet(deployNetworkDefinition);

//Get the blob for network output/iteration count for later testing
caffe2::TensorCPU& networkOutput = *workspace.GetBlob(network.GetOutputBlobNames()[0])->GetMutable<caffe2::TensorCPU>();

caffe2::TensorCPU& iter = *workspace.GetBlob("AdamSolver_iteration_iterator")->GetMutable<caffe2::TensorCPU>();

//Train the network
int64_t numberOfTrainingIterations = 100000;

for(int64_t iteration = 0; iteration < numberOfTrainingIterations; iteration++)
{
//Shuffle data every epoc through
if((iteration % trainingInputs.size()) == 0)
{
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);
}

//Load data into blobs
memcpy(inputBlob.mutable_data<float>(), &trainingInputs[iteration % trainingInputs.size()], inputBlob.nbytes());
memcpy(expectedOutputBlob.mutable_data<float>(), &trainingExpectedOutputs[iteration % trainingExpectedOutputs.size()], expectedOutputBlob.nbytes());

//Run network with loaded instance
trainingNetwork->Run();

}

//Output deploy results
{
std::ofstream pretrainedDeployResults("postTrainingDeployResults.csv");
for(int64_t iteration = 0; iteration < trainingInputs.size(); iteration++)
{
//Load data into blobs to csv for viewing
memcpy(inputBlob.mutable_data<float>(), &trainingInputs[iteration % trainingInputs.size()], inputBlob.nbytes());

deployNetwork->Run();

pretrainedDeployResults << *inputBlob.mutable_data<float>() << ", " << *networkOutput.mutable_data<float>() << ", " <<  *iter.mutable_data<int64_t>() << std::endl;
}
}

return 0;
} 
