// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util_xarch.hpp"
#include "util.hpp"

void ov::npuw::util::XARCH::unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& to) {
              unpack_impl(from, to);
            }

void ov::npuw::util::XARCH::unpack1(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to) {
              unpack1_impl(from, scale, to);
            }

void ov::npuw::util::XARCH::unpack2(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& zerop,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to) {
              unpack2_impl(from, zerop, scale, to);
            }

void ov::npuw::util::XARCH::to_f16(ov::Tensor& t) {
    to_f16_impl(t);
}