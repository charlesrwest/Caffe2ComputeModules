#include "UtilityFunctions.hpp"

#include<cstdlib>
#include "caffe2/core/operator_gradient.h"

using namespace GoodBot;

/**
This function generates a (capitalized) hex string using rand() to select digits.
@inputLength:How many digits the string should have
@return: The generated string
*/
std::string GoodBot::GenerateRandomHexString(int64_t inputLength)
{
const static std::string lookUpTable("0123456789ABCDEF");

std::string result;

while(result.size() < inputLength)
{
result.push_back(lookUpTable[rand() % lookUpTable.size()]);
}

return result;
}

std::string GoodBot::MakeGradientOperatorBlobName(const std::string& inputOperatorBlobName)
{
return inputOperatorBlobName + "_grad";
}

std::vector<caffe2::OperatorDef> GoodBot::GetGradientOperatorsFromOperator(const caffe2::OperatorDef& inputOperator)
{
std::vector<caffe2::GradientWrapper> gradientBlobNames;
for(int64_t outputIndex = 0; outputIndex < inputOperator.output_size(); outputIndex++)
{
caffe2::GradientWrapper wrapper;
wrapper.dense_ = MakeGradientOperatorBlobName(inputOperator.output(outputIndex));

gradientBlobNames.emplace_back(wrapper);
}

caffe2::GradientOpsMeta operatorsAndWrappers = caffe2::GetGradientForOp(inputOperator, gradientBlobNames);

for(caffe2::OperatorDef& gradientOperator : operatorsAndWrappers.ops_)
{
gradientOperator.set_is_gradient_op(true); //Mark as gradient operator
}

return operatorsAndWrappers.ops_;
}
