#include "FullyConnectedModuleDefinition.hpp"

#include<iostream>

using namespace GoodBot;

FullyConnectedModuleDefinition::FullyConnectedModuleDefinition(const FullyConnectedModuleDefinitionParameters& inputParameters) : moduleName(inputParameters.moduleName)
{
//Construct layers for this module
FullyConnectedLayerDefinitionParameters layerParameters;
layerParameters.numberOfInputs = inputParameters.numberOfInputs;
layerParameters.weightFillType = inputParameters.weightFillType;
layerParameters.biasFillType = inputParameters.biasFillType;
layerParameters.activationType = inputParameters.activationType;

for(int64_t layerIndex = 0; layerIndex < inputParameters.numberOfNodesInLayers.size(); layerIndex++)
{
int64_t numberOfNodesInLayer = inputParameters.numberOfNodesInLayers[layerIndex];

if(layerIndex == 0)
{
layerParameters.inputBlobName = inputParameters.inputBlobName;
}
else
{
layerParameters.inputBlobName = layers[layerIndex - 1].GetOutputBlobName();
}

layerParameters.numberOfNodes = numberOfNodesInLayer;
layerParameters.layerName = Name() + "_layer" + std::to_string(layerIndex);

layers.emplace_back(layerParameters);
}
}

const std::string& FullyConnectedModuleDefinition::Type() const
{
static const std::string type = "FullyConnectedModuleDefinition";

return type;
}

const std::string& FullyConnectedModuleDefinition::Name() const
{
return moduleName;
}

std::vector<std::string> FullyConnectedModuleDefinition::GetDeployInputBlobNames() const
{
return layers.front().GetDeployInputBlobNames();
}

std::vector<std::string> FullyConnectedModuleDefinition::GetDeployOutputBlobNames() const
{
return layers.back().GetDeployOutputBlobNames();
}

std::vector<std::string> FullyConnectedModuleDefinition::GetTrainingGradientBlobNames() const
{
std::vector<std::string> results;

for(std::vector<FullyConnectedLayerDefinition>::const_reverse_iterator iter = layers.rbegin(); iter != layers.rend(); iter++)
{
std::vector<std::string> layerGradientBlobNames = iter->GetTrainingGradientBlobNames();

results.insert(results.end(), layerGradientBlobNames.begin(), layerGradientBlobNames.end());
}
 
return results;
}

std::vector<caffe2::OperatorDef> FullyConnectedModuleDefinition::GetDeployNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;

for(const FullyConnectedLayerDefinition& layer : layers)
{
std::vector<caffe2::OperatorDef> layerOperators = layer.GetDeployNetworkOperators();

results.insert(results.end(), layerOperators.begin(), layerOperators.end());
}

return results;
}

std::vector<caffe2::OperatorDef> FullyConnectedModuleDefinition::GetTrainingNetworkInitializationOperators() const
{
std::vector<caffe2::OperatorDef> results;

for(const FullyConnectedLayerDefinition& layer : layers)
{
std::vector<caffe2::OperatorDef> layerOperators = layer.GetTrainingNetworkInitializationOperators();

results.insert(results.end(), layerOperators.begin(), layerOperators.end());
}


return results;
}
