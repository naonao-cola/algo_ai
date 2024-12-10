#if USE_TRT


#include "../../../include/private/trt/trt_common/trt_utils.h"
namespace TRT {

template <typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr)
{
    return std::shared_ptr<_T>(ptr, [](_T* p) { p->destroy(); });
}

std::vector<unsigned char> load_file(const std::string& file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

}  // namespace TRT


#endif //USE_TRT