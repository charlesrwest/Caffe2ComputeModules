#include "AveragedLossLayerDefinition.hpp"
#include "UtilityFunctions.hpp"

#include<iostream>

using namespace GoodBot;

AveragedLossLayerDefinition::AveragedLossLayerDefinition(const AveragedLossLayerDefinitionParameters& inputParameters) : inputBlobName(inputParameters.inputBlobName), layerName(inputParameters.layerName)
{
}

std::string AveragedLossLayerDefinition::Name() const
{
return layerName;
}

std::vector<std::string> AveragedLossLayerDefinition::GetInputBlobNames() const
{
return {inputBlobName};
}

std::vector<std::string> AveragedLossLayerDefinition::GetOutputBlobNames() const
{
//Passthrough if deploy mode
if(Mode() == "DEPLOY")
{
return {inputBlobName}; //Pass through
}

return {GetAveragedLossOutputBlobName()};
}

std::vector<caffe2::OperatorDef> AveragedLossLayerDefinition::GetNetworkOperators() const
{
if((Mode() == "TRAIN") || (Mode() == "TEST"))
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& averageLossOperator = results.back();
averageLossOperator.set_type("AveragedLoss");
averageLossOperator.add_input(inputBlobName);
averageLossOperator.add_output(GetAveragedLossOutputBlobName());

return results;
}

return {};
}


std::vector<caffe2::OperatorDef> AveragedLossLayerDefinition::GetGradientOperators() const
{

if(Mode() == "TRAIN")
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& averageLossLoopBackOperator = results.back();

averageLossLoopBackOperator.set_type("ConstantFill");
averageLossLoopBackOperator.add_input(GetAveragedLossOutputBlobName());
averageLossLoopBackOperator.add_output(MakeGradientOperatorBlobName(GetAveragedLossOutputBlobName()));
averageLossLoopBackOperator.set_is_gradient_op(true);
caffe2::Argument& argument = *averageLossLoopBackOperator.add_arg();
argument.set_name("value");
argument.set_f(1.0);

std::vector<caffe2::OperatorDef> normalGradientOperators = ComputeModuleDefinition::GetGradientOperators();

results.insert(results.end(), normalGradientOperators.begin(), normalGradientOperators.end());

return results;
}

return {};
}

std::string AveragedLossLayerDefinition::GetAveragedLossOutputBlobName() const
{
return Name() + "_averaged_loss";
}
