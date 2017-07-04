#include "L2LossLayerDefinition.hpp"
#include "UtilityFunctions.hpp"

using namespace GoodBot;

L2LossLayerDefinition::L2LossLayerDefinition(const L2LossLayerDefinitionParameters& inputParameters) : inputBlobName(inputParameters.inputBlobName), layerName(inputParameters.layerName), trainingExpectedOutputBlobName(inputParameters.trainingExpectedOutputBlobName), testExpectedOutputBlobName(inputParameters.testExpectedOutputBlobName)
{
}

std::string L2LossLayerDefinition::Name() const
{
return layerName;
}

std::vector<std::string> L2LossLayerDefinition::GetInputBlobNames() const
{
if(mode == "TRAIN")
{
return {inputBlobName, trainingExpectedOutputBlobName};
}
else if(mode == "TEST")
{
return {inputBlobName, testExpectedOutputBlobName};
}

return {};
}

std::vector<std::string> L2LossLayerDefinition::GetOutputBlobNames() const
{
if((mode == "TRAIN") || (mode == "TEST"))
{
return {GetL2LossOutputBlobName()};
}

return {};
}

std::vector<caffe2::OperatorDef> L2LossLayerDefinition::GetNetworkOperators() const
{
if(mode == "TRAIN")
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& lossOperator = results.back();

lossOperator.set_type("SquaredL2Distance");
lossOperator.add_input(trainingExpectedOutputBlobName);
lossOperator.add_input(inputBlobName);
lossOperator.add_output(GetL2LossOutputBlobName());

return results;
}
else if(mode == "TEST")
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& lossOperator = results.back();

lossOperator.set_type("SquaredL2Distance");
lossOperator.add_input(testExpectedOutputBlobName);
lossOperator.add_input(inputBlobName);
lossOperator.add_output(GetL2LossOutputBlobName());

return results;
}

return {};
}

std::string L2LossLayerDefinition::GetL2LossOutputBlobName() const
{
return Name() + "_L2_Loss";
}
