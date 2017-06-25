#include<iostream>
#include<string>
#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/logging.h"

#include "caffe2/core/workspace.h"



//Can store/manage blobs and nets in Workspace class, can also be used to run a plan

/**
This program expects two directory paths to be passed in:
Arg1: directory containing folders mnist-train-nchw-leveldb and mnist-test-nchw-leveldb
Arg2: directory containing network definitions to use
Arg3: output directory for generated files
*/
int main(int argc, char **argv)
{

//Get folders to access resources
if(argc < 4)
{
std::cerr << "Insufficient arguments:" << std::endl;
std::cerr << "Required:" << std::endl;
std::cerr << "Directory path containing folders mnist-train-nchw-leveldb and mnist-test-nchw-leveldb" << std::endl;
std::cerr << "Directory path containing network definitions (.pbtxt)" << std::endl;
std::cerr << "Directory path to place generated files" << std::endl;
return 1;
}

std::string training_data_folder(argv[1]);
std::string networkDefinitionFolder(argv[2]);
std::string output_folder(argv[3]);

//Initialize workspace which all data/calculation blobs will reside in
caffe2::Workspace workspace(output_folder);

//Initialization networks don't actually do any computations.  They just define/initialize the blobs that will be used in the other network definitions (shared and living in the workspace).

//This is why they are created and then run once before the other networks are.  The training initialization network fills all of the weights with their initialization values, while the test initialization network just loads the databases to test with without touching the weights

//Retrieve the protobuf object which defines the initialization values for the network from a text file
caffe2::NetDef trainingInitializationNetworkDefinition;

if(!ReadProtoFromFile(networkDefinitionFolder + "/train_init_net.pbtxt", &trainingInitializationNetworkDefinition))
{
std::cerr << "Error opening initialization network definition" << std::endl;
return 1;
}

//Load the protobuf object with training network architecture definition from a text file.  This network references the same blobs in the workspace that are set with the initialization network.
caffe2::NetDef  trainingNetworkDefinition;

if(!ReadProtoFromFile(networkDefinitionFolder + "/train_net.pbtxt", &trainingNetworkDefinition))
{
std::cerr << "Error opening training network definition" << std::endl;
return 1;
}

//Run the initialization network so that the blobs that the training network relies on are loaded and set with the initial values.
if(!workspace.RunNetOnce(trainingInitializationNetworkDefinition))
{
std::cerr << "Error loading/running training network initialization" << std::endl;
return 1;
}

//Define the training network in the workspace based on the training network protobuf object
caffe2::NetBase* trainingNetworkPointer = workspace.CreateNet(trainingNetworkDefinition);
if(trainingNetworkPointer == nullptr)
{
std::cerr << "Error instancing training network" << std::endl;
return 1;
}

//Train the network for 200 interations and report the accuracy at each step
int32_t numberOfTrainingRuns = 200;

for(int32_t runIndex = 0; runIndex < numberOfTrainingRuns; runIndex++)
{
//Run an iteration of the training network
workspace.RunNet(trainingNetworkDefinition.name());

//Retrieve the reported "accuracy" so that it can be printed out (1x1)
const caffe2::Blob* accuracyBlobPointer = workspace.GetBlob("accuracy");
if(accuracyBlobPointer == nullptr)
{
std::cerr << "Error, unable to access training accuracy blob" << std::endl;
return 1;
}

std::cout << "Accuracy: " <<  *(accuracyBlobPointer->Get<caffe2::Tensor<caffe2::CPUContext>>().data<float>()) << std::endl;

//Retrieve the reported loss so that it can be printed out (1x1)
const caffe2::Blob* lossBlobPointer = workspace.GetBlob("loss");
if(lossBlobPointer == nullptr)
{
std::cerr << "Error, unable to access training loss blob" << std::endl;
return 1;
}

std::cout << "Loss: " <<  *(lossBlobPointer->Get<caffe2::Tensor<caffe2::CPUContext>>().data<float>()) << std::endl;

//Both accuracy and loss are tensors
}

//Report success
std::cout << "All operations completed without error" << std::endl;

return 0;
} 
