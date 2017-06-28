#include "FullyConnectedNetworkArchitecture.hpp"
#include "FullyConnectedModuleDefinition.hpp"
#include "SoftMaxLayerDefinition.hpp"

#include<iostream>

using namespace GoodBot;

FullyConnectedNetworkArchitecture::FullyConnectedNetworkArchitecture(const std::string& inputArchitectureName, const FullyConnectedNetworkParameters& inputParameters) : name(inputArchitectureName), inputBlobName(inputParameters.inputBlobName), trainingExpectedOutputBlobName(inputParameters.trainingExpectedOutputBlobName),  testExpectedOutputBlobName(inputParameters.testExpectedOutputBlobName), nodesPerLayer(inputParameters.nodesPerLayer), weightFillType(inputParameters.weightFillType), biasFillType(inputParameters.biasFillType), numberOfInputs(inputParameters.numberOfInputs), activationType(inputParameters.activationType), lossType(inputParameters.lossType)
{
FullyConnectedModuleDefinitionParameters fullyConnectedParameters;
fullyConnectedParameters.inputBlobName = inputBlobName;
fullyConnectedParameters.numberOfInputs = numberOfInputs;
fullyConnectedParameters.numberOfNodesInLayers = nodesPerLayer;
fullyConnectedParameters.moduleName = Name() + "_fully_connected_module";
fullyConnectedParameters.weightFillType = weightFillType;
fullyConnectedParameters.biasFillType = biasFillType;
fullyConnectedParameters.activationType = activationType;

layers.emplace_back(new FullyConnectedModuleDefinition(fullyConnectedParameters));

SoftMaxLayerDefinitionParameters softMaxParameters;
softMaxParameters.inputBlobName = layers.front()->GetDeployOutputBlobNames().front(); //Should only be one output
softMaxParameters.layerName = Name() + "_soft_max";
softMaxParameters.trainingExpectedOutputBlobName = trainingExpectedOutputBlobName;
softMaxParameters.testExpectedOutputBlobName = testExpectedOutputBlobName;

layers.emplace_back(new SoftMaxLayerDefinition(softMaxParameters));
}

const std::string& FullyConnectedNetworkArchitecture::Name() const
{
return name;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTrainingInitializationNetwork() const
{
caffe2::NetDef initializationNetwork;

initializationNetwork.set_name(GetTrainingInitializationNetworkName());

for(const std::unique_ptr<ComputeModuleDefinition>& module : layers)
{
std::vector<caffe2::OperatorDef> initializationOperators = module->GetTrainingNetworkInitializationOperators();

for(const caffe2::OperatorDef& operatorDefinition : initializationOperators)
{
*initializationNetwork.add_op() = operatorDefinition; //Add to network
}
}

return initializationNetwork;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTrainingNetwork() const
{
caffe2::NetDef trainingNetwork;
trainingNetwork.set_name(GetTrainingNetworkName());

for(const std::unique_ptr<ComputeModuleDefinition>& module : layers)
{
std::vector<caffe2::OperatorDef> trainingOperators = module->GetTrainingNetworkOperators();

for(const caffe2::OperatorDef& operatorDefinition : trainingOperators)
{
*trainingNetwork.add_op() = operatorDefinition; //Add to network
}
}

//Add gradients
for(std::vector<std::unique_ptr<ComputeModuleDefinition>>::const_reverse_iterator iter = layers.rbegin(); iter != layers.rend(); iter++)
{
std::vector<caffe2::OperatorDef> gradientOperators = (*iter)->GetTrainingGradientOperators();

for(const caffe2::OperatorDef& operatorDefinition : gradientOperators)
{
*trainingNetwork.add_op() = operatorDefinition; //Add to network
}
}

return trainingNetwork;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTestNetwork() const
{
caffe2::NetDef testNetwork;
testNetwork.set_name(GetTestNetworkName());

for(const std::unique_ptr<ComputeModuleDefinition>& module : layers)
{
std::vector<caffe2::OperatorDef> testOperators = module->GetTestNetworkOperators();

for(const caffe2::OperatorDef& operatorDefinition : testOperators)
{
*testNetwork.add_op() = operatorDefinition; //Add to network
}
}

return testNetwork;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetDeployNetwork() const
{
caffe2::NetDef deployNetwork;
deployNetwork.set_name(GetDeployNetworkName());

for(const std::unique_ptr<ComputeModuleDefinition>& module : layers)
{
std::vector<caffe2::OperatorDef> deployOperators = module->GetDeployNetworkOperators();

for(const caffe2::OperatorDef& operatorDefinition : deployOperators)
{
*deployNetwork.add_op() = operatorDefinition; //Add to network
}
}

return deployNetwork;
}

int64_t FullyConnectedNetworkArchitecture::GetNumberOfFullyConnectedLayers() const
{
return nodesPerLayer.size();
}

