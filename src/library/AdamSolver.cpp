#include "AdamSolver.hpp"

#include "AdamModule.hpp"
#include "IteratorLayerDefinition.hpp"
#include "UtilityFunctions.hpp"
#include<iostream>

using namespace GoodBot;


AdamSolver::AdamSolver(const AdamSolverParameters& inputParameters) : learningRateBlobName(inputParameters.learningRateBlobName), iteratorBlobName(inputParameters.iteratorBlobName), learningRate(inputParameters.learningRate)
{
SetName(inputParameters.moduleName);

if(inputParameters.learningRateBlobName == "")
{
createLearningRateBlob = true;
learningRateBlobName = Name() + "_constant_learning_rate";
}

if(iteratorBlobName == "")
{
std::cout << "Adding iterator operator" << std::endl;
IteratorLayerDefinitionParameters iteratorParameters;
iteratorParameters.layerName = Name() + "_iteration";
iteratorParameters.initialValue = 0;

AddModule(*(new IteratorLayerDefinition(iteratorParameters)));

iteratorBlobName = modules.back()->GetOutputBlobNames()[0];
}
std::cout << "Module has " << modules.size() << " sub-modules" << std::endl;

AdamModuleParameters adamParameters;
adamParameters.learningRateBlobName = GetLearningRateBlobName();
adamParameters.iteratorBlobName = GetIteratorBlobName();
adamParameters.beta1 = inputParameters.beta1;
adamParameters.beta2 = inputParameters.beta2;
adamParameters.epsilon = inputParameters.epsilon;

std::cout << "Given " << inputParameters.trainableParameterNames.size() << " trainable parameters" << std::endl;

for(int64_t trainableBlobIndex = 0; trainableBlobIndex < inputParameters.trainableParameterNames.size(); trainableBlobIndex++)
{
const std::string& trainableParameter = inputParameters.trainableParameterNames[trainableBlobIndex];

adamParameters.moduleName = Name() + "_" + trainableParameter;
adamParameters.blobToUpdateName = trainableParameter;
adamParameters.gradientBlobName = MakeGradientOperatorBlobName(trainableParameter);
adamParameters.blobToUpdateShape = inputParameters.trainableParameterShapes[trainableBlobIndex];

AddModule(*(new AdamModule(adamParameters)));
}
}

std::vector<caffe2::OperatorDef> AdamSolver::GetNetworkInitializationOperators() const
{
std::vector<caffe2::OperatorDef> results;

results = CompositeComputeModuleDefinition::GetNetworkInitializationOperators();

if(Mode() == "TRAIN")
{
if(createLearningRateBlob)
{
results.emplace_back();
caffe2::OperatorDef& learningRateInitOperator = results.back();

learningRateInitOperator.set_type("ConstantFill");
learningRateInitOperator.add_output(GetLearningRateBlobName());
caffe2::Argument& shape = *learningRateInitOperator.add_arg();
shape.set_name("shape");
shape.add_ints(1);
caffe2::Argument& value = *learningRateInitOperator.add_arg();
value.set_name("value");
value.set_f(learningRate);
}
}

return results;
}

std::string AdamSolver::GetLearningRateBlobName() const
{
return learningRateBlobName;
}

std::string AdamSolver::GetIteratorBlobName() const
{
return iteratorBlobName;
}


