#include "AveragedL2LossModuleDefinition.hpp"

#include "L2LossLayerDefinition.hpp"
#include "AveragedLossLayerDefinition.hpp"
#include<iostream>

using namespace GoodBot;

AveragedL2LossModuleDefinition::AveragedL2LossModuleDefinition(const AveragedL2LossModuleDefinitionParameters& inputParameters)
{
struct AveragedL2LossModuleDefinitionParameters
{
std::string inputBlobName;
std::string moduleName;
std::string trainingExpectedOutputBlobName;
std::string testExpectedOutputBlobName;
};
SetName(inputParameters.moduleName);

L2LossLayerDefinitionParameters L2LossParameters;
L2LossParameters.inputBlobName = inputParameters.inputBlobName; //Should only be one output
L2LossParameters.layerName = Name() + "_L2_loss";
L2LossParameters.trainingExpectedOutputBlobName = inputParameters.trainingExpectedOutputBlobName;
L2LossParameters.testExpectedOutputBlobName = inputParameters.testExpectedOutputBlobName;

AddModule(*(new L2LossLayerDefinition(L2LossParameters)));

SetMode("TRAIN");

AveragedLossLayerDefinitionParameters averagedLossParameters;
averagedLossParameters.inputBlobName = modules.back()->GetOutputBlobNames().front(); //Should only be one output
averagedLossParameters.layerName = Name() + "_averaged_loss";

AddModule(*(new AveragedLossLayerDefinition(averagedLossParameters)));

SetMode("TRAIN");
}
