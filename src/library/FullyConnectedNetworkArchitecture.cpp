#include "FullyConnectedNetworkArchitecture.hpp"
#include "FullyConnectedModuleDefinition.hpp"
#include "SoftMaxLayerDefinition.hpp"
#include "L2LossLayerDefinition.hpp"
#include "AveragedLossLayerDefinition.hpp"

#include<iostream>

/*
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

layers.front()->SetMode("TRAIN");

if(lossType == "SoftmaxWithLoss")
{
SoftMaxLayerDefinitionParameters softMaxParameters;
softMaxParameters.inputBlobName = layers.front()->GetOutputBlobNames().front(); //Should only be one output
softMaxParameters.layerName = Name() + "_soft_max";
softMaxParameters.trainingExpectedOutputBlobName = trainingExpectedOutputBlobName;
softMaxParameters.testExpectedOutputBlobName = testExpectedOutputBlobName;

layers.emplace_back(new SoftMaxLayerDefinition(softMaxParameters));
}
else if(lossType == "SquaredL2Distance")
{
L2LossLayerDefinitionParameters L2LossParameters;
L2LossParameters.inputBlobName = layers.front()->GetOutputBlobNames().front(); //Should only be one output
L2LossParameters.layerName = Name() + "_L2_loss";
L2LossParameters.trainingExpectedOutputBlobName = trainingExpectedOutputBlobName;
L2LossParameters.testExpectedOutputBlobName = testExpectedOutputBlobName;

layers.emplace_back(new L2LossLayerDefinition(L2LossParameters));
layers.back()->SetMode("TRAIN");


AveragedLossLayerDefinitionParameters averagedLossParameters;
averagedLossParameters.inputBlobName = layers.back()->GetOutputBlobNames().front(); //Should only be one output
averagedLossParameters.layerName = Name() + "_averaged_loss";

layers.emplace_back(new AveragedLossLayerDefinition(averagedLossParameters));
layers.back()->SetMode("TRAIN");
}

}

const std::string& FullyConnectedNetworkArchitecture::Name() const
{
return name;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTrainingInitializationNetwork()
{
caffe2::NetDef initializationNetwork;

initializationNetwork.set_name(GetTrainingInitializationNetworkName());

SetMode("TRAIN");
for(const std::unique_ptr<ComputeModuleDefinition>& module : layers)
{
std::vector<caffe2::OperatorDef> initializationOperators = module->GetNetworkInitializationOperators();

for(const caffe2::OperatorDef& operatorDefinition : initializationOperators)
{
*initializationNetwork.add_op() = operatorDefinition; //Add to network
}
}

return initializationNetwork;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTrainingNetwork()
{
caffe2::NetDef trainingNetwork;
trainingNetwork.set_name(GetTrainingNetworkName());

SetMode("TRAIN");
for(const std::unique_ptr<ComputeModuleDefinition>& module : layers)
{
std::vector<caffe2::OperatorDef> trainingOperators = module->GetNetworkOperators();

for(const caffe2::OperatorDef& operatorDefinition : trainingOperators)
{
*trainingNetwork.add_op() = operatorDefinition; //Add to network
}
}

//Add gradients
for(std::vector<std::unique_ptr<ComputeModuleDefinition>>::const_reverse_iterator iter = layers.rbegin(); iter != layers.rend(); iter++)
{
std::vector<caffe2::OperatorDef> gradientOperators = (*iter)->GetGradientOperators();

for(const caffe2::OperatorDef& operatorDefinition : gradientOperators)
{
*trainingNetwork.add_op() = operatorDefinition; //Add to network
}
}

return trainingNetwork;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTestNetwork()
{
caffe2::NetDef testNetwork;
testNetwork.set_name(GetTestNetworkName());

SetMode("TEST");
for(const std::unique_ptr<ComputeModuleDefinition>& module : layers)
{
std::vector<caffe2::OperatorDef> testOperators = module->GetNetworkOperators();

for(const caffe2::OperatorDef& operatorDefinition : testOperators)
{
*testNetwork.add_op() = operatorDefinition; //Add to network
}
}

return testNetwork;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetDeployNetwork()
{
caffe2::NetDef deployNetwork;
deployNetwork.set_name(GetDeployNetworkName());

SetMode("DEPLOY");
for(const std::unique_ptr<ComputeModuleDefinition>& module : layers)
{
std::vector<caffe2::OperatorDef> deployOperators = module->GetNetworkOperators();

for(const caffe2::OperatorDef& operatorDefinition : deployOperators)
{
*deployNetwork.add_op() = operatorDefinition; //Add to network
}
}

return deployNetwork;
}

bool FullyConnectedNetworkArchitecture::SetMode(const std::string& inputMode)
{
for(std::unique_ptr<ComputeModuleDefinition>& modulePtr : layers)
{
modulePtr->SetMode(inputMode);
}
}

int64_t FullyConnectedNetworkArchitecture::GetNumberOfFullyConnectedLayers() const
{
return nodesPerLayer.size();
}

*/
