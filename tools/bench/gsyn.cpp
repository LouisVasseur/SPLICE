// Synthetic key sets with a known answer, for the period discovery: writes count + sorted keys.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
using u64 = uint64_t;
int main(int, char** v) {
  const std::string fam = v[1];
  const u64 n = std::strtoull(v[2], nullptr, 10);
  std::mt19937_64 g(9);
  auto U = [&](double a, double b) { return a + (b - a) * std::generate_canonical<double, 53>(g); };
  std::vector<u64> k;
  k.reserve(n);
  const u64 P = 100000, P2 = 77777, P3 = 123457, rows = n / 100;
  if (fam == "rows") { for (u64 i = 0; i < n; ++i) k.push_back((u64(1) << 30) + g() % rows * P + u64(U(0, 0.1 * double(P)))); }
  else if (fam == "three_blocks") {
    u64 base = u64(1) << 30;
    for (auto [p, share] : {std::pair<u64, double>{P, 0.4}, {P2, 0.3}, {P3, 0.3}}) {
      const u64 m = u64(double(n) * share), r = m / 100;
      for (u64 i = 0; i < m; ++i) k.push_back(base + g() % r * p + u64(U(0, 0.1 * double(p))));
      base += (r + 1) * p;
    }
  }
  else if (fam == "triangle") { for (u64 i = 0; i < n; ++i) { const u64 row = g() % rows; const u64 s = row * P / rows; k.push_back((u64(1) << 30) + row * P + s + g() % (P - s)); } }
  else if (fam == "rows_and_noise") { const u64 span = rows * P; for (u64 i = 0; i < n; ++i) k.push_back((g() & 1) ? (u64(1) << 30) + g() % rows * P + u64(U(0, 0.1 * double(P))) : (u64(1) << 30) + g() % span); }
  else if (fam == "jitter") { std::vector<u64> st(rows); u64 x = u64(1) << 30; for (auto& s : st) { x += u64(U(0.95, 1.05) * double(P)); s = x; } for (u64 i = 0; i < n; ++i) k.push_back(st[g() % rows] + u64(U(0, 0.1 * double(P)))); }
  else if (fam == "soft_wave") { const double span = double(rows) * double(P); while (k.size() < n) { const double x = U(0, span); if (U(0, 1.5) < 1 + 0.5 * std::sin(2 * M_PI * x / double(P))) k.push_back((u64(1) << 30) + u64(x)); } }
  else if (fam == "grid2") { for (u64 i = 0; i < n; ++i) k.push_back((u64(1) << 30) + g() % 3000 * 40000000 + g() % 400 * P + u64(U(0, 0.1 * double(P)))); }
  else if (fam == "uniform") { for (u64 i = 0; i < n; ++i) k.push_back(g() >> 8); }
  else if (fam == "lognormal") { std::lognormal_distribution<double> d(0, 2); for (u64 i = 0; i < n; ++i) k.push_back(u64(d(g) * 1e9)); }
  else if (fam == "stride") { for (u64 i = 0; i < n; ++i) k.push_back(12345 + i * 65536); }
  else if (fam == "clusters") { for (int c = 0; c < 1000; ++c) { const u64 b = g() >> 4; for (u64 i = 0; i < n / 1000; ++i) k.push_back(b + g() % 10000000); } }
  std::sort(k.begin(), k.end());
  k.erase(std::unique(k.begin(), k.end()), k.end());
  FILE* f = std::fopen(v[3], "wb");
  const u64 m = k.size();
  std::fwrite(&m, 8, 1, f);
  std::fwrite(k.data(), 8, m, f);
  std::fclose(f);
  return 0;
}
