#pragma once

//#include "NetworkArchitecture.hpp"
#include "ComputeModuleDefinition.hpp"

#include<memory>

/*
namespace GoodBot
{

struct FullyConnectedNetworkParameters
{
std::string inputBlobName;
int64_t numberOfInputs;
std::string trainingExpectedOutputBlobName;
std::string testExpectedOutputBlobName;
std::vector<int64_t> nodesPerLayer;
std::string weightFillType = "XavierFill";
std::string biasFillType = "ConstantFill";
std::string activationType = "Sigmoid";
std::string lossType = "SoftmaxWithLoss";

int64_t batchSize = 1;
};

class FullyConnectedNetworkArchitecture : public NetworkArchitecture
{
public:
FullyConnectedNetworkArchitecture(const std::string& inputArchitectureName, const FullyConnectedNetworkParameters& inputParameters);

virtual const std::string& Name() const override;

virtual caffe2::NetDef GetTrainingInitializationNetwork() const override;

virtual caffe2::NetDef GetTrainingNetwork() const override;

virtual caffe2::NetDef GetTestNetwork() const override;

virtual caffe2::NetDef GetDeployNetwork() const override;

bool SetMode(const std::string& inputMode);

int64_t GetNumberOfFullyConnectedLayers() const;

protected:
std::string name;
std::string inputBlobName;
int64_t numberOfInputs;
std::string trainingExpectedOutputBlobName;
std::string testExpectedOutputBlobName;
std::vector<int64_t> nodesPerLayer;
std::string weightFillType;
std::string biasFillType;
std::string activationType;
std::string lossType;

std::vector<std::unique_ptr<ComputeModuleDefinition>> layers;
};




























} 
*/
