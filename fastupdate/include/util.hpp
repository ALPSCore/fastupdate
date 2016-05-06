#pragma once

#include <algorithm>
#include <iterator>

namespace alps {
  namespace fastupdate {

    //return permutation from time-ordering (1 or -1)
    template <typename InputIterator>
    int permutation(InputIterator begin, InputIterator end) {
      using std::swap;
      typedef typename std::iterator_traits<InputIterator>::value_type my_value_type;

      std::vector<my_value_type> values;
      std::copy(begin, end, std::back_inserter(values));

      const int N = values.size();
      int perm = 1;
      while (true) {
        bool exchanged = false;
        for (int i=0; i<N-1; ++i) {
          if ( !(values[i]<values[i+1]) ) {
            swap(values[i], values[i+1]);
            perm *= -1;
            exchanged = true;
          }
        }
        if (!exchanged) break;
      }
      return perm;
    }

  }
}
