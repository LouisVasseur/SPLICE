#pragma once
#include "types.hpp"
#include <bit>
#include <cstddef>
#include <limits>
#include <span>

#if !defined(__SIZEOF_INT128__)
#error "SCALE-LI requires GCC/Clang unsigned __int128 for exact integer interpolation."
#endif
namespace scaleli {
using U128 = unsigned __int128;
using I128 = __int128;
enum class Codec : std::uint8_t { Raw = 0, For = 1, Delta = 2, Linear = 3 };
inline const char* codec_name(Codec c) {
    switch(c) { case Codec::Raw: return "raw"; case Codec::For: return "for";
        case Codec::Delta: return "delta"; case Codec::Linear: return "linear"; }
    throw std::invalid_argument("invalid codec");
}
inline Codec parse_codec(const std::string& s) {
    for (auto c : {Codec::Raw, Codec::For, Codec::Delta, Codec::Linear})
        if (s == codec_name(c)) return c;
    throw std::invalid_argument("unknown codec: " + s);
}
inline void put_le(std::vector<std::uint8_t>& b, std::size_t off, std::uint64_t x, unsigned n) {
    if (off + n > b.size()) b.resize(off + n);
    for (unsigned j = 0; j < n; ++j) b[off + j] = std::uint8_t(x >> (8*j));
}
inline std::uint64_t get_le(std::span<const std::uint8_t> b, std::size_t off, unsigned n) {
    if (n > 8 || off > b.size() || n > b.size() - off) throw std::out_of_range("truncated block");
    std::uint64_t x = 0;
    for (unsigned j = 0; j < n; ++j) x |= std::uint64_t(b[off+j]) << (8*j);
    return x;
}
inline void put_bits(std::vector<std::uint8_t>& b, std::size_t bit, unsigned width, std::uint64_t v) {
    unsigned done = 0;
    while (done < width) {
        const unsigned off = unsigned(bit % 8), take = std::min(8u-off, width-done);
        const auto mask = (1u << take) - 1;
        b.at(bit/8) |= std::uint8_t(((v >> done) & mask) << off);
        done += take; bit += take;
    }
}
inline std::uint64_t get_bits(std::span<const std::uint8_t> b, std::size_t bit, unsigned width,
                              QueryStats* s = nullptr) {
    if (width > 64 || bit > b.size()*8 || width > b.size()*8-bit)
        throw std::out_of_range("invalid bit-packed access");
    unsigned done = 0; std::uint64_t v = 0;
    while (done < width) {
        const unsigned off = unsigned(bit % 8), take = std::min(8u-off, width-done);
        v |= std::uint64_t((b[bit/8] >> off) & ((1u << take)-1)) << done;
        if (s) { ++s->codec_bytes_examined; s->note(&b[bit/8]); }
        done += take; bit += take;
    }
    return v;
}
inline void put_varint(std::vector<std::uint8_t>& b, std::uint64_t v) {
    while (v >= 128) { b.push_back(std::uint8_t(v) | 0x80); v >>= 7; }
    b.push_back(std::uint8_t(v));
}
inline std::uint64_t get_varint(std::span<const std::uint8_t> b, std::size_t& pos, QueryStats* s=nullptr) {
    std::uint64_t v = 0;
    for (unsigned shift = 0; shift <= 63; shift += 7) {
        if (pos >= b.size()) throw std::out_of_range("truncated varint");
        const auto x = b[pos++];
        if (s) { ++s->codec_bytes_examined; s->note(&b[pos-1]); }
        if (shift == 63 && (x & 0xfe)) throw std::invalid_argument("overflowing varint");
        v |= std::uint64_t(x & 0x7f) << shift;
        if (!(x & 0x80)) return v;
    }
    throw std::invalid_argument("unterminated varint");
}
struct EncodedBlock {
    Codec codec = Codec::Raw;
    std::vector<std::uint8_t> bytes;
    // Transient training positions: NOT retained in the index.
    // Fractional bytes represent the first bit of a bit-packed key code.
    std::vector<double> key_positions;
};
inline std::uint64_t interpolation(std::uint64_t span, std::size_t i, std::size_t n) {
    return n <= 1 ? 0 : std::uint64_t((U128(span) * i) / (n - 1));
}
inline EncodedBlock encode_block(std::span<const Key> keys, Codec requested, unsigned restart=16) {
    if (keys.empty() || keys.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("invalid block count");
    if (!std::is_sorted(keys.begin(), keys.end())) throw std::invalid_argument("unsorted block");
    if (restart == 0 || restart > 65535) throw std::invalid_argument("invalid restart interval");
    EncodedBlock e; e.codec = requested; e.bytes.resize(16, 0); e.key_positions.resize(keys.size());
    auto& b=e.bytes; b[0]=std::uint8_t(requested);
    put_le(b,4,keys.size(),4); put_le(b,8,keys.front(),8);
    if (requested == Codec::Raw) {
        b.resize(16+keys.size()*8);
        for (std::size_t i=0;i<keys.size();++i) {
            put_le(b,16+i*8,keys[i],8); e.key_positions[i]=double(16+i*8);
        }
    } else if (requested == Codec::For) {
        const unsigned w=std::bit_width(keys.back()-keys.front()); b[1]=std::uint8_t(w);
        b.resize(16+(keys.size()*w+7)/8,0);
        for (std::size_t i=0;i<keys.size();++i) {
            put_bits(b,128+i*w,w,keys[i]-keys.front()); e.key_positions[i]=w ? 16+double(i*w)/8 : 8;
        }
    } else if (requested == Codec::Linear) {
        const auto span=keys.back()-keys.front(); I128 lo=0,hi=0;
        for (std::size_t i=0;i<keys.size();++i) {
            const I128 r=I128(keys[i]-keys.front())-interpolation(span,i,keys.size());
            lo=std::min(lo,r); hi=std::max(hi,r);
        }
        // A fallback is explicit; exact uint64 correctness is more important than compression.
        if (lo < std::numeric_limits<std::int64_t>::min() || hi > std::numeric_limits<std::int64_t>::max())
            return encode_block(keys,Codec::Raw,restart);
        const auto range=std::uint64_t(hi-lo); const unsigned w=std::bit_width(range); b[1]=std::uint8_t(w);
        put_le(b,16,span,8); put_le(b,24,std::bit_cast<std::uint64_t>(std::int64_t(lo)),8);
        b.resize(32+(keys.size()*w+7)/8,0);
        for (std::size_t i=0;i<keys.size();++i) {
            const I128 r=I128(keys[i]-keys.front())-interpolation(span,i,keys.size());
            put_bits(b,256+i*w,w,std::uint64_t(r-lo)); e.key_positions[i]=w ? 32+double(i*w)/8 : 16;
        }
    } else if (requested == Codec::Delta) {
        put_le(b,2,restart,2);
        const auto groups=(keys.size()+restart-1)/restart; b.resize(16+4*groups,0);
        for (std::size_t i=0;i<keys.size();++i) {
            e.key_positions[i]=double(b.size());
            if (i%restart==0) {
                if (b.size()>std::numeric_limits<std::uint32_t>::max()) throw std::length_error("block offset overflow");
                put_le(b,16+4*(i/restart),b.size(),4); put_le(b,b.size(),keys[i],8);
            } else put_varint(b,keys[i]-keys[i-1]);
        }
    } else throw std::invalid_argument("invalid codec");
    return e;
}
inline std::size_t block_count(std::span<const std::uint8_t> b) { return get_le(b,4,4); }
// Cache lines of a byte range the decoder reads. Called only from inside an existing
// "if (s)" guard, so nothing is added to the uninstrumented path. Every range here is
// at most 16 bytes, so it spans at most two lines; it still delegates to
// QueryStats::note_range, which covers EVERY line of the range, so the bound is a
// property of the code rather than of this comment and a longer range added later
// cannot silently drop the lines between its endpoints.
inline void note_range(QueryStats* s, std::span<const std::uint8_t> b, std::size_t off, std::size_t n) {
    if (!n || off >= b.size() || n > b.size() - off) return;
    s->note_range(&b[off], n);
}
inline Key key_at(std::span<const std::uint8_t> b, std::size_t i, QueryStats* s=nullptr) {
    const auto n=block_count(b); if (i>=n) throw std::out_of_range("key index");
    // Header bytes 0..15: codec tag, width, restart, count, base. Read on every key_at.
    if (s) { ++s->key_at_calls; note_range(s,b,0,16); }
    const auto codec=Codec(b[0]); const auto base=get_le(b,8,8);
    if (codec==Codec::Raw) {
        if(s){++s->decoded_keys;s->codec_bytes_examined+=8;note_range(s,b,16+i*8,8);} return get_le(b,16+i*8,8);
    }
    if (codec==Codec::For) {
        const auto r=get_bits(b,128+i*b[1],b[1],s);
        if(r>std::numeric_limits<Key>::max()-base) throw std::invalid_argument("FOR overflow");
        if(s)++s->decoded_keys; return base+r;
    }
    if (codec==Codec::Linear) {
        if(s)note_range(s,b,16,16); // span and lo
        const auto span=get_le(b,16,8); const auto lo=std::bit_cast<std::int64_t>(get_le(b,24,8));
        const auto d=get_bits(b,256+i*b[1],b[1],s);
        const I128 k=I128(base)+interpolation(span,i,n)+lo+I128(d);
        if(k<0 || k>I128(std::numeric_limits<Key>::max())) throw std::invalid_argument("linear reconstruction overflow");
        if(s)++s->decoded_keys; return Key(k);
    }
    if(codec==Codec::Delta) {
        const auto r=get_le(b,2,2); if(!r) throw std::invalid_argument("zero restart");
        if(s)note_range(s,b,16+4*(i/r),4); // restart-group offset
        std::size_t pos=get_le(b,16+4*(i/r),4); auto k=get_le(b,pos,8); pos+=8;
        if(s){++s->decoded_keys;s->codec_bytes_examined+=8;note_range(s,b,pos-8,8);}
        for(std::size_t j=(i/r)*r;j<i;++j) {
            const auto d=get_varint(b,pos,s);
            if(d>std::numeric_limits<Key>::max()-k) throw std::invalid_argument("delta overflow");
            k+=d; if(s)++s->decoded_keys;
        }
        return k;
    }
    throw std::invalid_argument("invalid codec tag");
}
inline std::vector<Key> decode_block(std::span<const std::uint8_t> b, QueryStats* s=nullptr) {
    const auto n=block_count(b);
    if(s)note_range(s,b,0,16);
    // Serialized blocks are internal. Bound corrupt counts before allocating.
    if(n>16*1024*1024) throw std::length_error("invalid block size");
    std::vector<Key> keys(n);
    if(Codec(b[0])!=Codec::Delta) {
        for(std::size_t i=0;i<n;++i)keys[i]=key_at(b,i,s);
    } else {
        const auto r=get_le(b,2,2); if(!r)throw std::invalid_argument("zero restart");
        std::size_t pos=0;
        for(std::size_t i=0;i<n;++i) {
            if(i%r==0){pos=get_le(b,16+4*(i/r),4);keys[i]=get_le(b,pos,8);pos+=8;
                if(s){s->codec_bytes_examined+=8;note_range(s,b,16+4*(i/r),4);note_range(s,b,pos-8,8);}}
            else {
                const auto d=get_varint(b,pos,s);
                if(d>std::numeric_limits<Key>::max()-keys[i-1]) throw std::invalid_argument("delta overflow");
                keys[i]=keys[i-1]+d;
            }
            if(s)++s->decoded_keys;
        }
    }
    if(!std::is_sorted(keys.begin(),keys.end()))throw std::invalid_argument("decoded keys not sorted");
    return keys;
}
// Byte coordinate of the first encoded bit; this is NOT an address safe to dereference.
inline double encoded_position(std::span<const std::uint8_t> b,std::size_t i) {
    if(i>=block_count(b)) throw std::out_of_range("position index");
    switch(Codec(b[0])) {
        case Codec::Raw: return double(16+8*i);
        case Codec::For: return b[1] ? 16+double(i*b[1])/8 : 8;
        case Codec::Linear: return b[1] ? 32+double(i*b[1])/8 : 16;
        case Codec::Delta: {
            const auto r=get_le(b,2,2); if(!r)throw std::invalid_argument("zero restart");
            std::size_t p=get_le(b,16+4*(i/r),4); if(i%r==0)return double(p);
            p+=8;
            for(std::size_t j=1;j<i%r;++j)(void)get_varint(b,p);
            return double(p);
        }
    }
    throw std::invalid_argument("invalid codec");
}
} // namespace scaleli
