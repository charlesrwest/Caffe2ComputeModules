#define CATCH_CONFIG_MAIN //Make main function automatically
#include "catch.hpp"
#include<cstdlib>
#include<string>


class testClass1
{
public:
int testArray[2];
};

class testClass2
{
public:
testClass2() : integerReference(testArray[0])
{

}

int testArray[2];
const int &integerReference;
};

void sayThings(std::string inputString, int inputDelayTime)
{
system(("espeak -s 100 -v mb-en1 \"" + inputString + "\"").c_str());

sleep(inputDelayTime);
}

TEST_CASE("Example test case", "[Example]")
{
SECTION("Example section", "[Example]")
{
/*
printf("Size of test class 1: %ld\n", sizeof(testClass1));
printf("Size of test class 2: %ld\n", sizeof(testClass2));
*/

/*
for(int i=0; i<100000; i++)
{
if(i % 10 == 0)
{
sayThings("Cookie", 1);
}
else
{
sayThings(std::to_string(i).c_str(), 1);
}
}
*/

char characterArray[5] = "HELO";

printf("%ld\n", ((int64_t *) characterArray)[0]);
printf("%lu\n", ((uint64_t *) characterArray)[0]);
printf("%lf\n", ((double *) characterArray)[0]);
}
}


