#include "FullyConnectedLayerDefinition.hpp"
#include "UtilityFunctions.hpp"

using namespace GoodBot;

FullyConnectedLayerDefinition::FullyConnectedLayerDefinition(const FullyConnectedLayerDefinitionParameters& inputParameters) : inputBlobName(inputParameters.inputBlobName), layerName(inputParameters.layerName), weightFillType(inputParameters.weightFillType), biasFillType(inputParameters.biasFillType), activationType(inputParameters.activationType), numberOfInputs(inputParameters.numberOfInputs), numberOfNodes(inputParameters.numberOfNodes)
{
}

const std::string& FullyConnectedLayerDefinition::Type() const
{
static const std::string type = "FullyConnectedLayerDefinition";

return type;
}

const std::string& FullyConnectedLayerDefinition::Name() const
{
return layerName;
}

std::vector<std::string> FullyConnectedLayerDefinition::GetDeployInputBlobNames() const
{
return {inputBlobName};
}

std::vector<std::string> FullyConnectedLayerDefinition::GetDeployOutputBlobNames() const
{
return {GetOutputBlobName()};
}

std::vector<std::string> FullyConnectedLayerDefinition::GetTrainingGradientBlobNames() const
{
return {MakeGradientOperatorBlobName(GetOutputBlobName()), MakeGradientOperatorBlobName(GetFullyConnectedOutputBlobName())};
}

std::vector<caffe2::OperatorDef> FullyConnectedLayerDefinition::GetDeployNetworkOperators() const
{
std::vector<caffe2::OperatorDef> result;

result.emplace_back();
caffe2::OperatorDef& fullyConnectedOperator = result.back();
fullyConnectedOperator.set_type("FC");
fullyConnectedOperator.add_input(inputBlobName);
fullyConnectedOperator.add_input(GetWeightsBlobName());
fullyConnectedOperator.add_input(GetBiasesBlobName());
fullyConnectedOperator.add_output(GetFullyConnectedOutputBlobName());

//Add activation layer
result.emplace_back();
caffe2::OperatorDef& activationOperator = result.back();
activationOperator.set_type(activationType);
activationOperator.add_input(GetFullyConnectedOutputBlobName());
activationOperator.add_output(GetOutputBlobName());

return result;
}

std::vector<caffe2::OperatorDef> FullyConnectedLayerDefinition::GetTrainingNetworkInitializationOperators() const
{
std::vector<caffe2::OperatorDef> result;

//Setup weights/biases
result.emplace_back();
caffe2::OperatorDef& weightOperator = result.back();
weightOperator.set_type(weightFillType);
caffe2::Argument& weightShape = *weightOperator.add_arg();
weightShape.set_name("shape");
weightShape.add_ints(numberOfNodes); //Number of nodes in this layer
weightShape.add_ints(numberOfInputs); //Number of inputs to this layer
weightOperator.add_output(GetWeightsBlobName());


result.emplace_back();
caffe2::OperatorDef& biasOperator = result.back();
biasOperator.set_type(biasFillType);
biasOperator.add_output(GetBiasesBlobName());
caffe2::Argument& biasShape = *biasOperator.add_arg();
biasShape.set_name("shape");
biasShape.add_ints(numberOfNodes); //Number of nodes in this layer


return result;
}


std::string FullyConnectedLayerDefinition::GetWeightsBlobName() const
{
return Name() + "_weights";
}

std::string FullyConnectedLayerDefinition::GetBiasesBlobName() const
{
return Name() + "_biases";
}

std::string FullyConnectedLayerDefinition::GetFullyConnectedOutputBlobName() const
{
return Name() + "_fully_connected";
}

std::string FullyConnectedLayerDefinition::GetOutputBlobName() const
{
return Name() + "_output";
}

int64_t FullyConnectedLayerDefinition::GetNumberOfNodes() const
{
return numberOfNodes;
}
