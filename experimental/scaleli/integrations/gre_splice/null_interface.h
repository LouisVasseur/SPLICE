// GRE harness baseline 'nullindex' (SPLICE, MIT): an O(1) index that touches no memory in get(), so a run of it
// measures GRE itself (op sampling, the operations array, the virtual call, the timer). Used only to subtract the
// harness from perf counters (SERVER.md 5b step 7) and as the harness residual (calibration item 5); never a result.
// get() reports every key as found with payload = key, so GRE's success_read equals the op count.
#pragma once
#include <cstddef>
#include <utility>
#include "../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class NullInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
  typedef std::pair<KEY_TYPE, PAYLOAD_TYPE> record_t;
  size_t n_ = 0;

public:
  void init(Param *param = nullptr) {}

  void bulk_load(record_t *key_value, size_t num, Param *param = nullptr) { n_ = num; }

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) { val = static_cast<PAYLOAD_TYPE>(key); return true; }

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) { return false; }

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) { return false; }

  bool remove(KEY_TYPE key, Param *param = nullptr) { return false; }

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, record_t *result, Param *param = nullptr) { return 0; }

  long long memory_consumption() { return 0; }
};
