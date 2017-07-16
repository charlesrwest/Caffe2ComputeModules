#include "IteratorLayerDefinition.hpp"
#include<iostream>

using namespace GoodBot;

IteratorLayerDefinition::IteratorLayerDefinition(const IteratorLayerDefinitionParameters& inputParameters) : initialValue(inputParameters.initialValue)
{
SetName(inputParameters.layerName);
}

std::vector<std::string> IteratorLayerDefinition::GetInputBlobNames() const
{
return {IteratorBlobName()};
}

std::vector<std::string> IteratorLayerDefinition::GetOutputBlobNames() const
{
return {IteratorBlobName()};
}

std::vector<caffe2::OperatorDef> IteratorLayerDefinition::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& iteratorOperator = results.back();

iteratorOperator.set_type("Iter");
iteratorOperator.add_input(IteratorBlobName());
iteratorOperator.add_output(IteratorBlobName());

return results;
}


std::vector<caffe2::OperatorDef> IteratorLayerDefinition::GetNetworkInitializationOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& initializationOperator = results.back();

initializationOperator.set_type("ConstantFill");
initializationOperator.add_output(IteratorBlobName());
caffe2::Argument& shape = *initializationOperator.add_arg();
shape.set_name("shape");
shape.add_ints(1);
caffe2::Argument& value = *initializationOperator.add_arg();
value.set_name("value");
value.set_i(0);
caffe2::Argument& dataType = *initializationOperator.add_arg();
dataType.set_name("dtype");
dataType.set_i(caffe2::TensorProto_DataType_INT64); 

return results;
}


std::string IteratorLayerDefinition::IteratorBlobName() const
{
return Name() + "_iterator";
}
