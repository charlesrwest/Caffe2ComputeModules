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

std::vector<caffe2::OperatorDef> GoodBot::ReorderOperatorsToResolveDependencies(const std::vector<caffe2::OperatorDef>& inputOperators, const std::vector<std::string>& inputExistingBlobNames)
{
//Inefficient, but doesn't really matter all that much at this scale
std::vector<caffe2::OperatorDef> results;

using v_size_t = std::vector<caffe2::OperatorDef>::size_type;

std::set<std::string> availableInputBlobNames;
availableInputBlobNames.insert(inputExistingBlobNames.begin(), inputExistingBlobNames.end());

std::set<v_size_t> processedEntryIndices;

v_size_t numberOfEntriesRemovedThisLoop = 1;  //Initialize to non-zero to prevent exit at start of loop

std::function<bool(const caffe2::OperatorDef&)> RequiredInputBlobsAreAvailable = [&](const caffe2::OperatorDef& inputOperator)
{
for(int64_t inputIndex = 0; inputIndex < inputOperator.input_size(); inputIndex++)
{
if(availableInputBlobNames.count(inputOperator.input(inputIndex)) == 0)
{
return false; //This operator requires an input that is not available yet
}
}

return true; //All input requirements are satisfied
};

std::function<void(v_size_t)> AddOperatorToResults = [&](v_size_t inputOperatorIndex)
{
const caffe2::OperatorDef& operatorToAdd = inputOperators[inputOperatorIndex];

results.emplace_back(operatorToAdd);
//Add associated outputs to the available blob pool
for(int64_t outputIndex = 0; outputIndex < operatorToAdd.output_size(); outputIndex++)
{
availableInputBlobNames.emplace(operatorToAdd.output(outputIndex));
}

processedEntryIndices.emplace(inputOperatorIndex);
numberOfEntriesRemovedThisLoop++;
};

while((numberOfEntriesRemovedThisLoop > 0) && (results.size() < inputOperators.size()))
{
numberOfEntriesRemovedThisLoop = 0;

for(v_size_t operatorIndex = 0; operatorIndex < inputOperators.size(); operatorIndex++)
{
const caffe2::OperatorDef& currentOperator = inputOperators[operatorIndex];

if(processedEntryIndices.count(operatorIndex) > 0)
{
continue; //We already have this entry, so skip
}

if(RequiredInputBlobsAreAvailable(currentOperator))
{
AddOperatorToResults(operatorIndex);
}
}

}

return results;
}
