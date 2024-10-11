// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "logging.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace util {

bool is_set(const std::size_t sub_idx, const std::string& opt);

// Every great project has its own string class...
// NB: Newer C++ standards would allow to use string views or smt
ov::Tensor tensor_from_const(const std::shared_ptr<ov::Node>& node);

bool starts_with(const std::string& str, const std::string& prefix);

std::string fmt(std::size_t number, std::size_t total);


void gather(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& idx, const ov::SoPtr<ov::ITensor>& dst);

using View = std::vector<std::size_t>;
ov::SoPtr<ov::ITensor> view(const ov::SoPtr<ov::ITensor>& src, const View& from, const View& to);

ov::SoPtr<ov::ITensor> view(const ov::SoPtr<ov::ITensor>& src, std::size_t dim, std::size_t offset, std::size_t len);

void to_f32(const ov::Tensor& in, ov::Tensor& out);
void transpose(ov::Tensor& t);
void permute(ov::Tensor& t, const std::vector<std::size_t>& axes);
ov::Tensor concat(const std::vector<ov::Tensor>& tt, std::size_t axis);

namespace at {
template <class M>
struct Impl {
    using V = typename M::mapped_type;

    M* m = nullptr;
    explicit Impl(M* pM) : m(pM) {}

    template <typename K>
    V& at(const K& k) {
        const auto iter = m->find(k);
        if (iter == m->end()) {
            std::stringstream ss;
            ss << "Key " << k << " is not found in a map of type " << typeid(m).name();
            const auto msg = ss.str();
            LOG_ERROR(msg);
            throw std::out_of_range(msg);
        }
        return iter->second;
    }

    template <typename K>
    const V& at(const K& k) const {
        return const_cast<Impl*>(this)->at(k);
    }
};

template <typename M>
Impl<M> _(M* pM) {
    return Impl<M>(pM);
}

template <typename M>
Impl<M> _(std::shared_ptr<M> pM) {
    return Impl<M>(pM.get());
}

}  // namespace at

}  // namespace util
}  // namespace npuw
}  // namespace ov
