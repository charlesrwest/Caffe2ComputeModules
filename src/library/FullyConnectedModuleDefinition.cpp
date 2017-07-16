#include "FullyConnectedModuleDefinition.hpp"
#include "FullyConnectedLayerDefinition.hpp"

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
layerParameters.inputBlobName = modules[layerIndex - 1]->GetOutputBlobNames()[0];
layerParameters.numberOfInputs = inputParameters.numberOfNodesInLayers[layerIndex-1];
}

layerParameters.numberOfNodes = numberOfNodesInLayer;
layerParameters.layerName = Name() + "_layer" + std::to_string(layerIndex);

AddModule(*(new FullyConnectedLayerDefinition(layerParameters)));
}

//In case of solver, etc modules being added
storedInputBlobNames = modules.front()->GetInputBlobNames();
storedOutputBlobNames = modules.back()->GetOutputBlobNames();
}

std::string FullyConnectedModuleDefinition::Name() const
{
return moduleName;
}

std::vector<std::string> FullyConnectedModuleDefinition::GetInputBlobNames() const
{
return storedInputBlobNames;
}

std::vector<std::string> FullyConnectedModuleDefinition::GetOutputBlobNames() const
{
return storedOutputBlobNames;
}
