#define CATCH_CONFIG_MAIN //Make main function automatically
#include "catch.hpp"
#include<cstdlib>
#include<string>

#include "caffe2/core/workspace.h"
#include "caffe2/core/tensor.h"
#include<google/protobuf/text_format.h>
#include "FullyConnectedNetworkArchitecture.hpp"
#include<iostream>
#include<cmath>
#include<cassert>

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

TEST_CASE("Test generated Fully Connected NetDefs", "[Example]")
{
int64_t numberOfTrainingExamples = 1000;
std::vector<float> trainingInputs;
std::vector<float> trainingExpectedOutputs;

for(int64_t trainingExampleIndex = 0; trainingExampleIndex < numberOfTrainingExamples; trainingExampleIndex++)
{
trainingInputs.emplace_back(trainingExampleIndex*2.0*PI/(numberOfTrainingExamples+1));
trainingExpectedOutputs.emplace_back(sin(trainingInputs.back()));
}
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);


caffe2::Workspace workspace;
caffe2::CPUContext context;

caffe2::TensorCPU& inputBlob = *workspace.CreateBlob("inputBlob")->GetMutable<caffe2::TensorCPU>();
inputBlob.Resize(1, 1);
inputBlob.mutable_data<float>();

caffe2::TensorCPU& expectedOutputBlob = *workspace.CreateBlob("trainingExpectedOutputBlobName")->GetMutable<caffe2::TensorCPU>();
expectedOutputBlob.Resize(1, 1);
expectedOutputBlob.mutable_data<int>();

GoodBot::FullyConnectedNetworkParameters networkParameters;
networkParameters.inputBlobName = "inputBlob";
networkParameters.numberOfInputs = 1;
networkParameters.trainingExpectedOutputBlobName = "trainingExpectedOutputBlobName";
networkParameters.nodesPerLayer = {100};

GoodBot::FullyConnectedNetworkArchitecture networkArchitecture("testNetwork", networkParameters);

SECTION("Test training network", "[networkArchitecture]")
{
std::function<void(const google::protobuf::Message&)> print = [&](const google::protobuf::Message& inputMessage)
{
std::string buffer;

google::protobuf::TextFormat::PrintToString(inputMessage, &buffer);

std::cout << buffer<<std::endl;
};

//Initialize the network
caffe2::NetDef trainingNetworkInitializationDefinition = networkArchitecture.GetTrainingInitializationNetwork();

print(trainingNetworkInitializationDefinition);

caffe2::NetBase* initializationNetwork = workspace.CreateNet(trainingNetworkInitializationDefinition);
initializationNetwork->Run();

caffe2::NetDef trainingNetworkDefinition = networkArchitecture.GetTrainingNetwork();

print(trainingNetworkDefinition);

caffe2::NetBase* trainingNetwork = workspace.CreateNet(trainingNetworkDefinition);

/*
//Train the network
for(int64_t iteration = 0; iteration < 1000000; iteration++)
{
//Shuffle data every run through
if((iteration % trainingInputs.size()) == 0)
{
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);
}

//Load data into blobs
memcpy(inputBlob.mutable_data<float>(), &trainingInputs[iteration % trainingInputs.size()], inputBlob->nbytes());
memcpy(expectedOutputBlob.mutable_data<float>(), &trainingExpectedOutputs[iteration % trainingExpectedOutputs.size()], expectedOutputBlob->nbytes());
*/
trainingNetwork->Run();
//}

}
}


