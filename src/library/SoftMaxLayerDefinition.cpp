#include "SoftMaxLayerDefinition.hpp"
#include "UtilityFunctions.hpp"

using namespace GoodBot;

SoftMaxLayerDefinition::SoftMaxLayerDefinition(const SoftMaxLayerDefinitionParameters& inputParameters) : inputBlobName(inputParameters.inputBlobName), layerName(inputParameters.layerName), trainingExpectedOutputBlobName(inputParameters.trainingExpectedOutputBlobName), testExpectedOutputBlobName(inputParameters.testExpectedOutputBlobName)
{
}

std::string SoftMaxLayerDefinition::Name() const
{
return layerName;
}

std::vector<std::string> SoftMaxLayerDefinition::GetInputBlobNames() const
{
if((mode == "TRAIN") || (mode == "TEST"))
{
return {inputBlobName, trainingExpectedOutputBlobName};
}

return {inputBlobName};
}

std::vector<std::string> SoftMaxLayerDefinition::GetOutputBlobNames() const
{
if((mode == "TRAIN") || (mode == "TEST"))
{
return {GetSoftMaxOutputBlobName(), GetTrainingLossOutputBlobName()};
}

return {GetSoftMaxOutputBlobName()};
}

std::vector<caffe2::OperatorDef> SoftMaxLayerDefinition::GetNetworkOperators() const
{
if(mode == "DEPLOY")
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& softMax = results.back();

softMax.set_type("Softmax");
softMax.add_input(inputBlobName);
softMax.add_output(GetSoftMaxOutputBlobName());

return results;
}
else if(mode == "TRAIN")
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
else if(mode == "TEST")
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

return {};
}

std::vector<caffe2::OperatorDef> SoftMaxLayerDefinition::GetGradientOperators() const
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


std::vector<caffe2::OperatorDef> normalGradientOperators = ComputeModuleDefinition::GetGradientOperators();

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
