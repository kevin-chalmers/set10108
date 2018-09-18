#pragma once

#include <string>

class SHA256
{
protected:
    static const uint32_t sha256_k[];
    static constexpr size_t SHA224_256_BLOCK_SIZE = (512/8);
    void transform(const unsigned char *message, size_t block_nb);
    size_t m_tot_len;
    size_t m_len;
    unsigned char m_block[2 * SHA224_256_BLOCK_SIZE];
    uint32_t m_h[8];
public:
    void init();
    void update(const unsigned char *message, size_t len);
    void final(unsigned char *digest);
    static constexpr size_t DIGEST_SIZE = (256/8);
};

std::string sha256(const std::string &input);