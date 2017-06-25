#pragma once

#include "NetworkArchitecture.hpp"
#include "FullyConnectedLayerDefinition.hpp"

namespace GoodBot
{

struct FullyConnectedNetworkParameters
{
std::string inputBlobName;
int64_t numberOfInputs;
std::string expectedOutputBlobName;
int64_t numberOfOutputs;
std::string deployOutputBlobName;
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

virtual caffe2::NetDef GetTestInitializationNetwork() const override;

virtual caffe2::NetDef GetTestNetwork() const override;

virtual caffe2::NetDef GetDeployNetwork() const override;

int64_t GetNumberOfFullyConnectedLayers() const;

protected:
std::string name;
std::string inputBlobName;
std::string expectedOutputBlobName;
std::vector<int64_t> nodesPerLayer;
std::string weightFillType;
std::string biasFillType;
int64_t numberOfInputs;
int64_t numberOfOutputs;
std::string activationType;
std::string lossType;

std::vector<std::unique_ptr<LayerDefinition>> layers;
};




























} 
