#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <deque>

using namespace std;

template <typename It>
size_t my_distance(It first, It last)
{
    size_t result = 0;
    while (first != last)
    {
        ++first;
        ++result;
    }
    return result;
}

int main()
{
    set<string> names{"jie", "outisjie", "jojo", "C++"};
    auto jie_iter = find(names.begin(), names.end(), "jie");
    auto jojo_iter = find(names.begin(), names.end(), "jojo");

    cout << my_distance(jie_iter, jojo_iter) << endl;
    cout << names << endl;
};