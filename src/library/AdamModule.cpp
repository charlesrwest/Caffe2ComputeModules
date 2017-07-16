#include "AdamModule.hpp"

using namespace GoodBot;

AdamModule::AdamModule(const AdamModuleParameters& inputParameters) : blobToUpdateName(inputParameters.blobToUpdateName), blobToUpdateShape(inputParameters.blobToUpdateShape), gradientBlobName(inputParameters.gradientBlobName), learningRateBlobName(inputParameters.learningRateBlobName), iteratorBlobName(inputParameters.iteratorBlobName), beta1(inputParameters.beta1), beta2(inputParameters.beta2), epsilon(inputParameters.epsilon)
{
SetName(inputParameters.moduleName);
}

std::vector<std::string> AdamModule::GetInputBlobNames() const
{
if(Mode() == "TRAIN")
{
return {blobToUpdateName, GetMoment1BlobName(), GetMoment2BlobName(), gradientBlobName, learningRateBlobName, iteratorBlobName};
}

return {};
}

std::vector<std::string> AdamModule::GetOutputBlobNames() const
{
if(Mode() == "TRAIN")
{
return {blobToUpdateName, GetMoment1BlobName(), GetMoment2BlobName()};
}

return {};
}

std::vector<caffe2::OperatorDef> AdamModule::GetNetworkOperators() const
{
if(Mode() == "TRAIN")
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& adamOperator = results.back();
adamOperator.set_type("Adam");

std::vector<std::string> inputNames = GetInputBlobNames();
for(const std::string& blobName : inputNames)
{
adamOperator.add_input(blobName);
}

std::vector<std::string> outputNames = GetOutputBlobNames();
for(const std::string& blobName : outputNames)
{
adamOperator.add_output(blobName);
}

caffe2::Argument& beta1Arg = *adamOperator.add_arg();
beta1Arg.set_name("beta1");
beta1Arg.set_f(beta1);

caffe2::Argument& beta2Arg = *adamOperator.add_arg();
beta2Arg.set_name("beta2");
beta2Arg.set_f(beta2);

caffe2::Argument& epsilonArg = *adamOperator.add_arg();
epsilonArg.set_name("epsilon");
epsilonArg.set_f(epsilon);

return results;
}

return {};
}

std::vector<caffe2::OperatorDef> AdamModule::GetNetworkInitializationOperators() const
{
if(Mode() == "TRAIN")
{
std::vector<caffe2::OperatorDef> results;

for(const std::string& momentName : {GetMoment1BlobName(), GetMoment2BlobName()})
{
results.emplace_back();
caffe2::OperatorDef& currentInitOperator = results.back();

currentInitOperator.set_type("ConstantFill");
currentInitOperator.add_output(momentName);
caffe2::Argument& shape = *currentInitOperator.add_arg();
shape.set_name("shape");
for(int64_t dimension : blobToUpdateShape)
{
shape.add_ints(dimension);
}

caffe2::Argument& value = *currentInitOperator.add_arg();
value.set_name("value");
value.set_f(0.0);
}

return results;
}

return {};
}

std::vector<caffe2::OperatorDef> AdamModule::GetGradientOperators() const
{
return {};
}

std::string AdamModule::GetMoment1BlobName() const
{
return Name() + "_moment1";
}

std::string AdamModule::GetMoment2BlobName() const
{
return Name() + "_moment2";
}
