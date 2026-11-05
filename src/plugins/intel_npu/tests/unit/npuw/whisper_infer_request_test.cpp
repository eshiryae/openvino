// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Request-level tests for the stage selection of WhisperInferRequest. External
// pipelines reset the npuw_stored_tokens_state variable state between audio
// windows and after language detection. A reset must route the next inference
// to prefill, also when the new prompt has only one token. Submodels are
// compiled through a factory that returns fake sub-requests: their infer()
// only zeroes outputs and counts the calls, so the tests observe which stage
// the request ran.

#include "whisper/whisper_infer_request.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "executor.hpp"
#include "llm_compiled_model.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace {

using ov::test::npuw::build_whisper_decoder_test_model;
using ov::test::npuw::NullPlugin;

struct StageCalls {
    std::size_t prefill = 0;
    std::size_t generate = 0;
};

class FakeSubCompiledModel;

class FakeSubInferRequest final : public ov::ISyncInferRequest {
public:
    FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model, std::size_t* counter);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        return ov::ISyncInferRequest::get_tensor(port);
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }
    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

private:
    std::size_t* m_counter;
};

class FakeSubCompiledModel final : public ov::npuw::ICompiledModel_v0 {
public:
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                         const std::shared_ptr<const ov::IPlugin>& plugin,
                         std::size_t* counter)
        : ov::npuw::ICompiledModel_v0(model, plugin),
          m_model(model),
          m_counter(counter) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        auto self = std::static_pointer_cast<const FakeSubCompiledModel>(shared_from_this());
        return std::make_shared<FakeSubInferRequest>(std::move(self), m_counter);
    }
    std::shared_ptr<ov::npuw::IBaseInferRequest> create_base_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::IAsyncInferRequest> wrap_async_infer_request(
        std::shared_ptr<ov::npuw::IBaseInferRequest>) const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                        intel_npu::make_executor("whisper_request_task", 1),
                                                        intel_npu::make_executor("whisper_request_callback", 1));
    }
    std::string submodel_device(std::size_t) const override {
        return "CPU";
    }
    std::size_t num_submodels() const override {
        // One CPU submodel, so init_pre_alloc_device() resolves to CPU staging
        // instead of requesting NPU remote tensors from the stub plugin.
        return 1;
    }
    std::shared_ptr<ov::npuw::weights::Bank> get_weights_bank() const override {
        return {};
    }
    void set_weights_bank(std::shared_ptr<ov::npuw::weights::Bank>) override {}
    void finalize_weights_bank() override {}
    void reconstruct_closure() override {}
    void serialize(std::ostream&, const ov::npuw::s11n::CompiledContext&) const override {}

private:
    std::shared_ptr<ov::Model> m_model;
    std::size_t* m_counter;
};

FakeSubInferRequest::FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model,
                                         std::size_t* counter)
    : ov::ISyncInferRequest(std::move(compiled_model)),
      m_counter(counter) {
    for (const auto& input : get_compiled_model()->inputs()) {
        ov::ISyncInferRequest::set_tensor(input,
                                          ov::get_tensor_impl(ov::Tensor(input.get_element_type(), input.get_shape())));
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        ov::ISyncInferRequest::set_tensor(
            output,
            ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
    }
}

void FakeSubInferRequest::infer() {
    if (m_counter != nullptr) {
        ++(*m_counter);
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        auto tensor = ov::ISyncInferRequest::get_tensor(output);
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Routes each submodel to the counter of its stage by the friendly name that
// LLMCompiledModel gives it ("..._prefill" or "..._kv...").
class StageCountingFactory {
public:
    explicit StageCountingFactory(StageCalls& calls) : m_calls(calls) {}

    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        return [this](const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap&) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            const auto& name = model->get_friendly_name();
            std::size_t* counter = nullptr;
            if (ends_with(name, "_prefill")) {
                counter = &m_calls.prefill;
            } else if (name.find("_kv") != std::string::npos) {
                counter = &m_calls.generate;
            }
            return std::make_shared<FakeSubCompiledModel>(model, plugin, counter);
        };
    }

private:
    StageCalls& m_calls;
};

ov::Tensor make_i64(std::initializer_list<size_t> shape, int64_t fill_value) {
    ov::Tensor tensor(ov::element::i64, ov::Shape(shape));
    std::fill_n(tensor.data<int64_t>(), tensor.get_size(), fill_value);
    return tensor;
}

class WhisperInferRequestTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
        StageCountingFactory factory(m_calls);
        ov::AnyMap props{{"NPUW_LLM", "YES"},
                         {"NPUW_WHISPER", "YES"},
                         {"NPUW_WHISPER_EOS_TOKEN", "42"},
                         {"NPUW_DEVICES", "CPU"},
                         {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                         {"NPUW_LLM_MIN_RESPONSE_LEN", "64"}};
        m_compiled = std::make_shared<ov::npuw::LLMCompiledModel>(build_whisper_decoder_test_model(),
                                                                  m_plugin,
                                                                  props,
                                                                  factory.make_factory());
        ASSERT_NE(m_compiled, nullptr);
        ASSERT_EQ(m_calls.prefill, 0u);
        ASSERT_EQ(m_calls.generate, 0u);

        m_request = std::make_shared<ov::npuw::WhisperInferRequest>(m_compiled);
    }

    ov::SoPtr<ov::IVariableState> stored_tokens_state() {
        for (const auto& state : m_request->query_state()) {
            if (state->get_name() == "npuw_stored_tokens_state") {
                return state;
            }
        }
        ADD_FAILURE() << "npuw_stored_tokens_state is not exposed by the request";
        return {};
    }

    // Same as ov::InferRequest::reset_state(), which the GenAI Whisper decoder calls.
    void reset_state() {
        for (const auto& state : m_request->query_state()) {
            state->reset();
        }
    }

    int64_t stored_tokens() {
        return stored_tokens_state()->get_state()->data<int64_t>()[0];
    }

    void run(size_t num_tokens) {
        const auto& inputs = m_request->get_inputs();
        const auto enc_port = ov::npuw::util::find_port_by_name(inputs, "encoder_hidden_states").value();
        const auto& enc_shape = enc_port.get_partial_shape();
        ov::Tensor encoder_hidden_states(ov::element::f32,
                                         ov::Shape{1,
                                                   static_cast<size_t>(enc_shape[1].get_length()),
                                                   static_cast<size_t>(enc_shape[2].get_length())});
        std::fill_n(encoder_hidden_states.data<float>(), encoder_hidden_states.get_size(), 0.f);

        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "input_ids").value(),
                              ov::get_tensor_impl(make_i64({1, num_tokens}, 1)));
        m_request->set_tensor(enc_port, ov::get_tensor_impl(encoder_hidden_states));
        m_request->infer();
    }

    StageCalls m_calls;
    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::npuw::LLMCompiledModel> m_compiled;
    std::shared_ptr<ov::npuw::WhisperInferRequest> m_request;
};

// The stored tokens state follows prefill and generate, so an external
// pipeline reads the real KV position.
TEST_F(WhisperInferRequestTest, StoredTokensStateFollowsPrefillAndGenerate) {
    run(2);
    EXPECT_EQ(m_calls.prefill, 1u);
    EXPECT_EQ(stored_tokens(), 2);

    run(1);
    EXPECT_EQ(m_calls.generate, 1u);
    EXPECT_EQ(stored_tokens(), 3);
}

// After reset_state() a one-token prompt (language detection, or the next
// window of an English-only model) must start a new prefill. Without the sync
// it ran generate on the KV cache of the previous inference.
TEST_F(WhisperInferRequestTest, ResetRoutesSingleTokenPromptToPrefill) {
    run(2);
    run(1);
    ASSERT_EQ(m_calls.prefill, 1u);
    ASSERT_EQ(m_calls.generate, 1u);

    reset_state();
    EXPECT_EQ(stored_tokens(), 0);

    run(1);
    EXPECT_EQ(m_calls.prefill, 2u);
    EXPECT_EQ(m_calls.generate, 1u);
    EXPECT_EQ(stored_tokens(), 1);

    // The new conversation continues with an ordinary generate step.
    run(1);
    EXPECT_EQ(m_calls.generate, 2u);
    EXPECT_EQ(stored_tokens(), 2);
}

// A one-token prompt on a new request is a prefill: there is no KV cache yet.
TEST_F(WhisperInferRequestTest, FirstSingleTokenPromptRunsPrefill) {
    run(1);
    EXPECT_EQ(m_calls.prefill, 1u);
    EXPECT_EQ(m_calls.generate, 0u);
    EXPECT_EQ(stored_tokens(), 1);
}

}  // namespace
