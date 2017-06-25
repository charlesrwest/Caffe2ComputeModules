#include "UtilityFunctions.hpp"

#include<cstdlib>

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
return inputOperatorBlobName + "_gradient";
}

std::vector<caffe2::OperatorDef> GoodBot::GetGradientOperatorsFromOperator(const caffe2::OperatorDef& inputOperator)
{
std::vector<caffe2::GradientWrapper> gradientBlobNames;
for(int64_t outputIndex = 0; outputIndex < wrappers.size(); outputIndex++)
{
gradientBlobNames.emplace_back(MakeGradientOperatorBlobName(inputOperator.output(outputIndex)));
}

caffe2::GradientOpsMeta operatorsAndWrappers = caffe2::GetGradientForOp(inputOperator, gradientBlobNames);

for(caffe2::OperatorDef& gradientOperator : operatorsAndWrappers.ops_)
{
gradientOperator.set_is_gradient_op(true); //Mark as gradient operator
}

return operatorsAndWrappers.ops_;
}
