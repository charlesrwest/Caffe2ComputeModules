#include "SoftMaxLayerDefinition.hpp"
#include "UtilityFunctions.hpp"

using namespace GoodBot;

SoftMaxLayerDefinition::SoftMaxLayerDefinition(const SoftMaxLayerDefinitionParameters& inputParameters) : inputBlobName(inputParameters.inputBlobName), layerName(inputParameters.layerName), trainingExpectedOutputBlobName(inputParameters.trainingExpectedOutputBlobName), testExpectedOutputBlobName(inputParameters.testExpectedOutputBlobName)
{
}

const std::string& SoftMaxLayerDefinition::Type() const
{
static const std::string type = "SoftMaxLayerDefinition";

return type;
}

const std::string& SoftMaxLayerDefinition::Name() const
{
return layerName;
}

std::vector<std::string> SoftMaxLayerDefinition::GetDeployInputBlobNames() const
{
return {inputBlobName};
}

std::vector<std::string> SoftMaxLayerDefinition::GetTrainingInputBlobNames() const
{
return {inputBlobName, trainingExpectedOutputBlobName};
}

std::vector<std::string> SoftMaxLayerDefinition::GetTestInputBlobNames() const
{
return {inputBlobName, testExpectedOutputBlobName};
}

std::vector<std::string> SoftMaxLayerDefinition::GetDeployOutputBlobNames() const
{
return {GetSoftMaxOutputBlobName()};
}

std::vector<std::string> SoftMaxLayerDefinition::GetTrainingOutputBlobNames() const
{
return {GetSoftMaxOutputBlobName(), GetTrainingLossOutputBlobName()};
}

std::vector<std::string> SoftMaxLayerDefinition::GetTrainingGradientBlobNames() const
{
return {MakeGradientOperatorBlobName(GetTrainingLossOutputBlobName())};
}

std::vector<caffe2::OperatorDef> SoftMaxLayerDefinition::GetDeployNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& softMax = results.back();

softMax.set_type("Softmax");
softMax.add_input(inputBlobName);
softMax.add_output(GetSoftMaxOutputBlobName());

return results;
}

std::vector<caffe2::OperatorDef> SoftMaxLayerDefinition::GetTrainingNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& softMaxWithLoss = results.back();

softMaxWithLoss.set_type("SoftmaxWithLoss");
softMaxWithLoss.add_input(inputBlobName);
softMaxWithLoss.add_input(trainingExpectedOutputBlobName);
softMaxWithLoss.add_output(GetSoftMaxOutputBlobName());
softMaxWithLoss.add_output(GetTrainingLossOutputBlobName());

return results;
}

std::vector<caffe2::OperatorDef> SoftMaxLayerDefinition::GetTestNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& softMaxWithLoss = results.back();

softMaxWithLoss.set_type("SoftmaxWithLoss");
softMaxWithLoss.add_input(inputBlobName);
softMaxWithLoss.add_input(testExpectedOutputBlobName);
softMaxWithLoss.add_output(GetSoftMaxOutputBlobName());
softMaxWithLoss.add_output(GetTestLossOutputBlobName());

return results;
}

std::vector<caffe2::OperatorDef> SoftMaxLayerDefinition::GetTrainingNetworkInitializationOperators() const
{
return std::vector<caffe2::OperatorDef>();
}

std::vector<caffe2::OperatorDef> SoftMaxLayerDefinition::GetTrainingGradientOperators() const
{
std::vector<caffe2::OperatorDef> results;

//connector
results.emplace_back();
caffe2::OperatorDef& gradient = results.back();
gradient.set_type("ConstantFill");

caffe2::Argument& argument = *gradient.add_arg();
argument.set_name("value");
argument.set_f(1.0);

gradient.add_input(GetTrainingLossOutputBlobName());
gradient.add_output(MakeGradientOperatorBlobName(GetTrainingLossOutputBlobName()));
gradient.set_is_gradient_op(true);


std::vector<caffe2::OperatorDef> normalGradientOperators = ComputeModuleDefinition::GetTrainingGradientOperators();

results.insert(results.end(), normalGradientOperators.begin(), normalGradientOperators.end());

return results;
}


std::string SoftMaxLayerDefinition::GetSoftMaxOutputBlobName() const
{
return Name()+"_soft_max";
}

std::string SoftMaxLayerDefinition::GetTrainingLossOutputBlobName() const
{
return Name()+"_training_loss";
}

std::string SoftMaxLayerDefinition::GetTestLossOutputBlobName() const
{
return Name()+"_test_loss";
}
