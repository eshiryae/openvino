// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "just_sync_infer_request.hpp"

#include <algorithm>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "compiled_model.hpp"
#include "logging.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "plugin.hpp"
#include "util.hpp"
#include "util_xarch.hpp"
#include "weights_bank.hpp"

ov::npuw::JustInferRequest::JustInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : IBaseInferRequest(compiled_model) {
    m_use_function_pipelining = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FUNCALL_ASYNC>();
    if (m_use_function_pipelining) {
        LOG_WARN("Function call pipelining is enabled for " << m_npuw_model->m_name
                                                            << ", expect a higher memory consumption");
        m_funcall_pipeline.resize(m_num_submodels);
    }

    m_spatial_io.resize(m_num_submodels);

    // Create infer requests
    // Preallocate funcall tensors & substitute function call requests
    bool failover_happened = false;
    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_INFO("Creating infer request for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];

        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            // no model & no funcall - optimized out, do nothing
            LOG_INFO("OPTIMIZED OUT");
            continue;
        }

        // FIXME: Shouldn't this be handled by the base class? (in create_tensor)
        // A special case for function calls
        if (comp_model_desc.replaced_by) {
            // Pre-allocate output tesnors for this function call
            const auto real_idx = comp_model_desc.replaced_by.value();
            auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
            auto& proto_comp_model = proto_comp_model_desc.compiled_model;

            const auto num_outputs = proto_comp_model->outputs().size();

            // Initialize the spatial IO placeholders, if required
            if (proto_comp_model_desc.spatial) {
                m_spatial_io[real_idx].inputs.resize(proto_comp_model_desc.param_base);
                m_spatial_io[real_idx].input_tails.resize(proto_comp_model_desc.param_base);
                m_spatial_io[real_idx].outputs.resize(num_outputs);
                m_spatial_io[real_idx].output_tails.resize(num_outputs);

                if (proto_comp_model_desc.spatial->tail_size) {
                    // Preallocate extra buffers for tail processing
                    // Note: these buffers are allocated to the entire NWAY (> tail_size)
                    for (auto&& p : proto_comp_model_desc.spatial->params) {
                        const auto& iport = proto_comp_model_desc.compiled_model->inputs()[p.idx];
                        m_spatial_io[real_idx].input_tails[p.idx] =
                            ov::get_tensor_impl(ov::Tensor(iport.get_element_type(), iport.get_shape()));
                    }
                    const auto num_outs = proto_comp_model_desc.compiled_model->outputs().size();
                    for (std::size_t out_idx = 0u; out_idx < num_outs; out_idx++) {
                        const auto& oport = proto_comp_model_desc.compiled_model->outputs()[out_idx];
                        m_spatial_io[real_idx].output_tails[out_idx] =
                            ov::get_tensor_impl(ov::Tensor(oport.get_element_type(), oport.get_shape()));
                    }
                }
            }  // if(spatial)

            for (size_t out_idx = 0; out_idx < num_outputs; out_idx++) {
                const auto& port = proto_comp_model->outputs()[out_idx];
                ov::Shape shape = port.get_shape();

                // If the subgraph is spatial, promote the output size to the full vector size
                if (proto_comp_model_desc.spatial) {
                    shape[proto_comp_model_desc.spatial->out_dim] = proto_comp_model_desc.spatial->range;
                }
                m_funcall_result[LinkFrom{i, out_idx}] =
                    ov::get_tensor_impl(ov::Tensor(port.get_element_type(), shape));
            }
            if (real_idx != i) {
                // If this function call is NOT the function body, do nothing here - the original
                // request will be used.
                LOG_INFO("REUSE " << real_idx);
                continue;
            }
        }  // if(replaced_by)

        // Special cases are handled -- so nothing to do here
        bool recompiled = false;
        auto rqs = create_infer_requests(i, m_use_function_pipelining ? 2 : 1, &recompiled);
        failover_happened |= recompiled;
        m_subrequests[i] = rqs.at(0);
        m_subrequest_devices[i] = *comp_model_desc.device_it;
        if (comp_model_desc.replaced_by && m_use_function_pipelining) {
            m_funcall_pipeline[i].subrequest = rqs.at(1);
        }

        LOG_INFO("DONE");
    }  // for(submodels)

    if (failover_happened) {
        LOG_INFO("Refined device distribution:");
        LOG_BLOCK();
        m_npuw_model->log_device_dist();
    }

    // Identify connections for the funcall pipeline, if needed
    if (m_use_function_pipelining) {
        LOG_INFO("Setting up the funcall pipeline...");
        LOG_BLOCK();
        std::vector<std::optional<std::size_t>> prevs(m_num_submodels);
        for (std::size_t i = 0; i < m_num_submodels; i++) {
            auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];
            if (comp_model_desc.replaced_by) {  // a function call..
                // Use real_id to accumulate information about
                // different functions
                const auto real_id = comp_model_desc.replaced_by.value();
                if (!prevs[real_id]) {  // ..met for a first time
                    LOG_INFO("Mark subgraph[" << i << "] as a head of pipeline...");
                    m_funcall_heads.push_back(i);
                } else {  // ..seen before
                    // Record that _this_ id follows the last known
                    // _prev_ if in the funcall pipeline
                    const auto prev_id = prevs[real_id].value();
                    LOG_INFO("Mark subgraph[" << i << "] as a successor of subraph[" << prev_id
                                              << "] in the function pipeline");
                    m_funcall_pipeline[prev_id].next = {i};
                }
                prevs[real_id] = {i};
            }  // if (replaced_by)
        }
    }  // if(function_pipelining)

    // Preallocate input tensors
    LOG_INFO("Preallocating input tensors...");
    for (size_t i = 0; i < m_npuw_model->inputs().size(); i++) {
        const auto& port = m_npuw_model->inputs()[i];
        m_input_tensors.push_back(ov::get_tensor_impl(ov::Tensor(port.get_element_type(), port.get_shape())));
        m_port_to_tensor[port] = TensorStorage{m_input_tensors.back(), true};
    }  // for(inputs)

    // Preallocate output tensors
    LOG_INFO("Preallocating output tensors...");
    for (size_t i = 0; i < m_npuw_model->outputs().size(); i++) {
        LOG_BLOCK();
        const auto& port = m_npuw_model->outputs()[i];
        LOG_INFO("Output " << i << " of " << m_npuw_model->outputs().size() << ": " << port);

        // FIXME: Yes, the CompiledModel::ToSubmodel == JustInferRequest::LinkFrom
        const auto& from_submodel = m_npuw_model->m_outputs_to_submodels_outputs.at(i);

        LOG_INFO("Produced by Subgraph[" << from_submodel.first << "] / " << from_submodel.second);
        auto funcall_result_iter = m_funcall_result.find(from_submodel);

        const auto& tensor =
            funcall_result_iter != m_funcall_result.end()
                ? funcall_result_iter->second  // Function calls have their tensors allocated, so just use one
                : ov::get_tensor_impl(ov::Tensor(port.get_element_type(), port.get_shape()));

        m_output_tensors.push_back(tensor);
        m_port_to_tensor[port] = TensorStorage{tensor, true};
    }
    connect_subrequests();

    // Build the parameter/result mapping {{{
    m_subrequests_gio.resize(m_subrequests.size());

    // Parameters: stage 1...
    for (size_t i = 0; i < m_npuw_model->inputs().size(); i++) {
        const auto& to_submodel = m_npuw_model->m_inputs_to_submodels_inputs.at(i);
        if (to_submodel != CompiledModel::NO_LINK) {
            std::size_t sub_idx{}, in_idx{};
            std::tie(sub_idx, in_idx) = to_submodel;
            m_subrequests_gio.at(sub_idx).global_params[i] = in_idx;
        }
    }  // for(inputs)

    // Parameters: stage 2...
    for (auto&& it : m_npuw_model->m_param_subscribers) {
        const auto param_idx = it.first;
        for (auto&& to_submodel : it.second) {
            std::size_t sub_idx{}, in_idx{};
            std::tie(sub_idx, in_idx) = to_submodel;
            m_subrequests_gio.at(sub_idx).global_params[param_idx] = in_idx;
        }
    }

    // Results
    for (size_t i = 0; i < m_npuw_model->outputs().size(); i++) {
        std::size_t sub_idx{}, out_idx{};
        std::tie(sub_idx, out_idx) = m_npuw_model->m_outputs_to_submodels_outputs.at(i);
        m_subrequests_gio.at(sub_idx).global_results[i] = out_idx;
    }
    // }}}

    // Sort out how to handle weights bank and unpack
    auto& wbank = m_npuw_model->m_weights_bank;
    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_VERB("Trying to preemptively set tensors for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];

        if (!comp_model_desc.compiled_model || !comp_model_desc.replaced_by) {
            continue;
        }

        const auto real_idx = comp_model_desc.replaced_by.value();
        auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

        auto& request = m_subrequests[real_idx];

        for (std::size_t cidx = 0u; cidx < comp_model_desc.closure.size(); cidx++) {
            const auto& closure = comp_model_desc.closure[cidx];

            const auto closure_param_id = comp_model_desc.param_base + cidx;
            const auto& iport = func_desc.compiled_model->inputs()[closure_param_id];

            // No update required to this tensor in runtime - so it can be set only once
            if (!comp_model_desc.update_required[cidx]) {
                request->set_tensor(iport, ov::get_tensor_impl(wbank->get(closure, *func_desc.device_it)));
            }
        }  // for(closure)
        LOG_VERB("DONE");
    }
}

void ov::npuw::JustInferRequest::connect_subrequests() {
    LOG_INFO("Connecting subrequests...");
    LOG_BLOCK();
    auto& subm = m_npuw_model->m_compiled_submodels;
    auto& subreqs = m_subrequests;
    for (const auto& kvp : m_npuw_model->m_submodels_input_to_prev_output) {
        const auto& subm_idx_to = kvp.first.first;
        const auto& port_idx_to = kvp.first.second;
        const auto& subm_idx_from = kvp.second.first;
        const auto& port_idx_from = kvp.second.second;

        LOG_DEBUG("Subgraph[" << subm_idx_from << "]/" << port_idx_from << " --> "
                              << "Subgraph[" << subm_idx_to << "]/" << port_idx_to);
        LOG_BLOCK();

        if (subm[subm_idx_from].replaced_by && subm[subm_idx_to].replaced_by) {
            // A function call to function call connection:
            // - Skip it here, setting in/out tensors will be handled in runtime
            LOG_DEBUG("Skip: both are function calls");
            continue;
        } else if (subm[subm_idx_from].replaced_by && !subm[subm_idx_to].replaced_by) {
            // A function call to normal subgraph connection:
            // - Take a tensor from the storage & assign it to the reader
            const auto& iport = m_subrequests[subm_idx_to]->get_compiled_model()->inputs()[port_idx_to];
            const auto& tensor = m_funcall_result.at(LinkFrom{subm_idx_from, port_idx_from});
            subreqs[subm_idx_to]->set_tensor(iport, tensor);
            LOG_DEBUG("Set Subgraph[" << subm_idx_to << "]/" << iport << " to internal tensor");
        } else if (!subm[subm_idx_from].replaced_by && subm[subm_idx_to].replaced_by) {
            LOG_DEBUG("Skip: reader is a function call");
            continue;
        } else if (!subreqs[subm_idx_from] && subreqs[subm_idx_to]) {
            // Subrequests may be optimized out, but in this case there should be
            // no connection between them and their consumers in the input_to_prev
            // map (links are erased & Parameters are replaced with Const) at earlier
            // stages.
            OPENVINO_THROW("FATAL: \"Prev. Output\" Request ",
                           subm_idx_from,
                           " in input_to_prev_output mapping was optimized out,"
                           " but it consumer request ",
                           subm_idx_to,
                           " wasn't!");
        } else if (!subreqs[subm_idx_to]) {
            // FIXME: Links like this probbaly shouldn't exist in the map, too.
            // Need to research why this is not THROW
            LOG_WARN("\"Input\" Request in input_to_prev_output mapping was optimized out");
            continue;
        } else {
            // Final case:
            NPUW_ASSERT(!subm[subm_idx_from].replaced_by);  // prod subgraph isn't a funcall
            NPUW_ASSERT(!subm[subm_idx_to].replaced_by);    // cons subgraph isn't a funcall
            NPUW_ASSERT(subreqs[subm_idx_from]);            // prod request is created
            NPUW_ASSERT(subreqs[subm_idx_to]);              // cons request is created

            // Just set one's output tensor to another's input
            const auto& iport = subreqs[subm_idx_to]->get_compiled_model()->inputs()[port_idx_to];
            const auto& oport = subreqs[subm_idx_from]->get_compiled_model()->outputs()[port_idx_from];
            const auto& tensor = subreqs[subm_idx_from]->get_tensor(oport);
            LOG_DEBUG("Set Subgraph[" << subm_idx_to << "]/" << iport << " to Subgraph[" << subm_idx_from << "]/"
                                      << oport);
            subreqs[subm_idx_to]->set_tensor(iport, tensor);
        }
    }  // for(map)
    LOG_INFO("Done");
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::JustInferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> variable_states = {};
    for (const auto& request : m_subrequests) {
        if (!request)  // optimized out
            continue;
        for (auto&& state : request->query_state()) {
            if (!state._so)
                state._so = request._so;
            variable_states.emplace_back(state);
        }
    }
    return variable_states;
}

std::vector<ov::ProfilingInfo> ov::npuw::JustInferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    for (size_t i = 0; i < m_subrequests.size(); ++i) {
        if (!m_subrequests[i])  // optimized out
            continue;
        auto&& subreq_info = m_subrequests[i]->get_profiling_info();
        for (auto&& rec : subreq_info)
            rec.node_name = std::string("subgraph") + std::to_string(i) + ": " + rec.node_name;
        info.insert(info.end(), subreq_info.begin(), subreq_info.end());
    }
    return info;
}

void ov::npuw::JustInferRequest::prepare_for_infer() {
    LOG_DEBUG("Preparing to infer...");
    LOG_BLOCK();

    // Submit global parameters (if needed) for the first subgraph
    bind_global_parameters(next(0));

    // If funcall pipelining is enabled, prefill the function "heads"
    // with constant arguments. The list of heads is empty otherwise.
    for (auto&& id : m_funcall_heads) {
        LOG_DEBUG("Pre-initializing weights for subgraph[" << id << "]");
        unpack_closure(id, m_subrequests[id]);
    }
    LOG_DEBUG("Done");
}

ov::npuw::IBaseInferRequest::RqPtr ov::npuw::JustInferRequest::get_real_subrequest(std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    return m_subrequests[real_idx];
}

bool ov::npuw::JustInferRequest::valid_subrequest(std::size_t idx) const {
    auto* ncthis = const_cast<ov::npuw::JustInferRequest*>(this);
    return ncthis->get_real_subrequest(idx) != nullptr;
}

void ov::npuw::JustInferRequest::start_subrequest(std::size_t idx) {
    m_subrequests[idx]->start_async();
}

void ov::npuw::JustInferRequest::bind_global_parameters(std::size_t idx) {
    LOG_DEBUG("Binding parameters for Subgraph[" << idx << "]");
    LOG_BLOCK();

    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);

    LOG_DEBUG("Real idx is..." << real_idx);

    const bool do_copy = needs_copy(idx);
    const auto& iodesc = m_subrequests_gio.at(idx);

    const auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    const bool is_spatial = proto_comp_model_desc.spatial.has_value();

    // a list of ports to copy tensors, if needed: FROM -> TO
    std::vector<std::pair<ov::SoPtr<ov::ITensor>, ov::Output<const ov::Node>>> copy_list;

    // pick which subrequest we actually work on here
    auto subr = [&]() {
        if (now_idx() && real_idx == real(now_idx().value()) && m_use_function_pipelining) {
            LOG_DEBUG("Accessing the pipeline subrequest");
            // The real index of request we need to prepare IS
            // the same request which executes now AND
            // function_pipelining enabled - select the reserve request.
            NPUW_ASSERT(m_funcall_pipeline[real_idx].subrequest);
            return m_funcall_pipeline[real_idx].subrequest;
        }
        // Otherwise: Just a return a subrequest which is in place.
        // If it is a function call and we have function pipelining ON,
        // it is still the right subrequest we can use.
        LOG_DEBUG("Accessing the primary subrequest");
        return m_subrequests[real_idx];
    }();

    // Check if the given subgraph's input is spatial
    auto is_spatial_param = [&](std::size_t sub_in_idx) -> bool {
        if (!is_spatial) {
            return false;  // Early return
        }
        auto& spatial = proto_comp_model_desc.spatial.value();
        return std::any_of(spatial.params.begin(), spatial.params.end(), [&](const auto& p) -> bool {
            return p.idx == sub_in_idx;
        });
    };

    for (auto&& it : iodesc.global_params) {
        std::size_t param_idx{}, sub_in_idx{};
        std::tie(param_idx, sub_in_idx) = it;
        LOG_DEBUG("Processing " << param_idx << " -> " << sub_in_idx << std::endl);

        const auto& g_port = m_npuw_model->inputs()[param_idx];
        const auto& g_tnsr = m_port_to_tensor.at(g_port).tensor;
        const auto& s_port = subr->get_inputs()[sub_in_idx];
        LOG_DEBUG("Processing " << g_port << " -> " << s_port << "...");
        LOG_BLOCK();
        if (!is_spatial_param(sub_in_idx)) {
            // Input parameter is non-spatial, do normal handling
            if (do_copy) {
                LOG_DEBUG("Will be copied");
                copy_list.emplace_back(g_tnsr, s_port);
            } else {
                LOG_DEBUG("Will be set");
                subr->set_tensor(s_port, g_tnsr);
            }
        } else {
            // Register for future use
            m_spatial_io[real_idx].inputs.at(sub_in_idx) = g_tnsr;
        }
    }

    LOG_DEBUG("Running copy...");
    ov::parallel_for(copy_list.size(), [&](std::size_t idx) {
        auto& it = copy_list[idx];
        ov::SoPtr<ov::ITensor> dst = subr->get_tensor(it.second);
        it.first->copy_to(dst._ptr);
    });

    // Run host-side gather, if required
    if (comp_model_desc.host_gather.dst_idx != -1) {
        auto& dst = comp_model_desc.closure[comp_model_desc.host_gather.dst_idx - comp_model_desc.param_base];
        const auto& vocab = comp_model_desc.closure[comp_model_desc.host_gather.src_idx - comp_model_desc.param_base];
        const auto& lport = comp_model_desc.compiled_model->inputs()[comp_model_desc.host_gather.idx_idx];
        const auto lookup = subr->get_tensor(lport);
        ov::npuw::util::gather(ov::get_tensor_impl(vocab), lookup, ov::get_tensor_impl(dst));
    }

    LOG_DEBUG("Done");
}

void ov::npuw::JustInferRequest::bind_global_results(std::size_t idx) {
    LOG_DEBUG("Binding results for Subgraph[" << idx << "]");
    LOG_BLOCK();

    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    if (comp_model_desc.replaced_by) {
        // Don't do here - function call will take the right tensor
        // itself. Note it may be implemented more efficently than now
        // (and in some cases, the tensor can be pre-set)
        LOG_DEBUG("Skipping this too now - function will do it for itself");
        return;
    }

    const auto& iodesc = m_subrequests_gio.at(idx);
    for (auto&& it : iodesc.global_results) {
        std::size_t result_idx{}, sub_out_idx{};
        std::tie(result_idx, sub_out_idx) = it;
        const auto& g_port = m_npuw_model->outputs()[result_idx];
        const auto& s_port = m_subrequests[idx]->get_outputs()[sub_out_idx];
        m_subrequests[idx]->set_tensor(s_port, m_port_to_tensor.at(g_port).tensor);
    }

    LOG_DEBUG("Done");
}

void ov::npuw::JustInferRequest::function_prologue(std::size_t idx) {
    LOG_DEBUG("Preparing Subgraph[" << idx << "] funcall prologue");
    LOG_BLOCK();

    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];

    NPUW_ASSERT(comp_model_desc.replaced_by);
    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

    const bool is_spatial = func_desc.spatial.has_value();

    // Function call prologue:
    // 1. Walk through function dependencies and set the respective tensors
    //    as parameters
    for (size_t i = 0; i < func_desc.param_base; i++) {
        LOG_DEBUG("Binding parameter[" << i << "]...");
        LOG_BLOCK();
        const auto& iport = func_desc.compiled_model->inputs()[i];

        auto link_iter = m_npuw_model->m_submodels_input_to_prev_output.find({idx, i});
        if (link_iter != m_npuw_model->m_submodels_input_to_prev_output.end()) {
            std::size_t prod_idx;
            std::size_t prod_port;
            std::tie(prod_idx, prod_port) = link_iter->second;

            if (!m_npuw_model->m_compiled_submodels[prod_idx].replaced_by) {
                // Producer is a normal model -> take its tensor directly
                const auto& oport = m_npuw_model->m_compiled_submodels[prod_idx].compiled_model->outputs()[prod_port];
                auto i_tensor = m_subrequests[prod_idx]->get_tensor(oport);
                if (!is_spatial) {
                    // Non-spatial case - set immediately
                    m_subrequests[real_idx]->set_tensor(iport, i_tensor);
                } else {
                    // Spatial case - defer
                    m_spatial_io[real_idx].inputs.at(i) = i_tensor;
                }
            } else {
                // Producer is a function - maybe the same as we're calling now.
                // Take its tensor from the storage
                auto i_tensor = m_funcall_result.at({prod_idx, prod_port});
                if (!is_spatial) {
                    // Non-spatial case - again, set immediately
                    m_subrequests[real_idx]->set_tensor(iport, m_funcall_result.at({prod_idx, prod_port}));
                } else {
                    // Spatial case - defer
                    m_spatial_io[real_idx].inputs.at(i) = i_tensor;
                }
            }
        }
    }  // for(param_base)

    // 2. Unpack the function closure -- right here, if pipelining if not enabled.
    // If it is enabled, the flow is a little bit different - see run_subrequest_for_success()
    // for details.
    if (!m_use_function_pipelining) {
        LOG_DEBUG("Unpacking closures...");
        LOG_BLOCK();
        unpack_closure(idx, m_subrequests[real_idx]);
    }

    // 3. Tell the function which results to produce (this time).
    // Note it covers both internal tensors used by other subgraphs as well as
    // the Result tensors for the entire network.
    // ..Since the tensors allocated for outputs of the networks ARE taken from the
    // "funcall_results" if those are produced by funcall results.
    for (std::size_t i = 0; i < func_desc.compiled_model->outputs().size(); i++) {
        LOG_DEBUG("Binding result[" << i << "]...");
        auto& oport = func_desc.compiled_model->outputs()[i];
        auto o_tensor = m_funcall_result.at({idx, i});
        if (!is_spatial) {
            // Non-spatial case - set immediately
            m_subrequests[real_idx]->set_tensor(oport, o_tensor);
        } else {
            // Spatial case - defer
            m_spatial_io[real_idx].outputs.at(i) = o_tensor;
        }
    }
    LOG_DEBUG("Done");
}

void ov::npuw::JustInferRequest::unpack_closure(std::size_t idx, RqPtr request) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];

    NPUW_ASSERT(comp_model_desc.replaced_by);
    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

    // Bind extra parameters from the function's closure
    // First, do easy things & delay heavy stuff
    std::vector<std::size_t> closure_unpack_required;
    std::vector<std::size_t> closure_copy_required;

    for (std::size_t cidx = 0u; cidx < comp_model_desc.closure.size(); cidx++) {
        auto& closure = comp_model_desc.closure[cidx];

        const auto closure_param_id = comp_model_desc.param_base + cidx;
        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];
        const auto& clparam = request->get_tensor(iport);
        if (closure.get_element_type() != clparam->get_element_type()) {
            // Remember where the unpack is required
            closure_unpack_required.push_back(cidx);
        } else if (comp_model_desc.update_required[cidx]) {
            if (needs_copy(idx)) {
                // Remember where copy is requried
                closure_copy_required.push_back(cidx);
            } else {
                // Easy case, just set one to another
                request->set_tensor(iport, ov::get_tensor_impl(closure));
            }
        }
    }  // for(closure)

    // m_ms_unpack += ov::npuw::perf::ms_to_run([&](){
    ov::parallel_for(closure_copy_required.size(), [&](std::size_t j) {
        auto cidx = closure_copy_required[j];
        auto& closure = comp_model_desc.closure[cidx];
        const auto closure_param_id = comp_model_desc.param_base + cidx;
        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];
        auto clparam = request->get_tensor(iport);
        ov::get_tensor_impl(closure)->copy_to(clparam._ptr);
    });
    // }); // ms_to_run

    for (std::size_t j = 0; j != closure_unpack_required.size(); j++) {
        // NB: No need to protect anything here as containers are all
        // preallocated and we only access elements under particular (thread
        // -local) indices.
        auto cidx = closure_unpack_required[j];

        // FIXME: zerops are stored with absolute indexing, this needs to be aligned
        auto& closure = comp_model_desc.closure[cidx];

        const auto closure_param_id = comp_model_desc.param_base + cidx;
        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];
        auto clparam = request->get_tensor(iport);

        if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx] && comp_model_desc.zerops[cidx]) {
            // Unpacking this weight requires scaling with zero points...
            ov::npuw::util::XARCH::unpack2(ov::get_tensor_impl(closure),
                                    ov::get_tensor_impl(comp_model_desc.zerops[cidx]),
                                    ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                    clparam);
        } else if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx]) {
            // Unpacking this weight requires scaling
            ov::npuw::util::XARCH::unpack1(ov::get_tensor_impl(closure),
                                   ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                   clparam);
        } else {
            // Unpacking this weight doesn't require scaling
            ov::npuw::util::XARCH::unpack(ov::get_tensor_impl(closure), clparam);
        }
    }
}

void ov::npuw::JustInferRequest::recreate_subrequests(std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto real_idx = comp_model_desc.replaced_by.value_or(idx);

    auto new_rqs = create_infer_requests(idx, m_use_function_pipelining ? 2 : 1);

    // NB: Regardless if this subrequest was a function call
    // or not, always use the real_idx here - for regular
    // subrequests, real_id == idx, but for function calls it
    // is critical here to update the function body, not the
    // function calls (which are left empty now in the vector)
    m_subrequests[real_idx] = new_rqs.at(0);
    if (comp_model_desc.replaced_by && m_use_function_pipelining) {
        m_funcall_pipeline[real_idx].subrequest = new_rqs.at(1);
    }
    // After an infer request is recreated, the internal cross-request
    // connections should be re-established (in/out tensors reset properly)
    // Note: these two proceduers do the full I/O reset procedure what's
    // overkill - only affected subrequest(s) could be updated instead,
    // but it is a more complex thing and can be implemented separately
    connect_subrequests();
    m_subrequest_devices[idx] = *comp_model_desc.device_it;
}

void ov::npuw::JustInferRequest::run_subrequest_for_success(std::size_t idx, bool& failover) {
    failover = false;
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto real_idx = comp_model_desc.replaced_by.value_or(idx);

    // Infer is also fail-safe...
    bool job_done = false;
    bool dump_in = false;
    bool next_prepared = false;
    while (!job_done) {
        bool should_recreate = false;
        if (m_subrequest_devices[real_idx] != *m_npuw_model->m_compiled_submodels[real_idx].device_it) {
            // This may happen when there's multiple NPUW's infer
            // requests created and some failure occurs in one of
            // those before another reaches this point.
            LOG_INFO("Recreating subrequest[" << real_idx << "] because model was recompiled for "
                                              << *m_npuw_model->m_compiled_submodels[real_idx].device_it << " device.");
            recreate_subrequests(real_idx);
        }

        // Feeding the global Parameters is now part of the common
        // execution pipeline: See how it is done in
        // `unsafe_run_this_prep_next()`.  Now we only need to bind
        // the subrequest' outputs to global Results, if relevant.
        bind_global_results(idx);

        if (comp_model_desc.replaced_by) {
            function_prologue(idx);
        }
        if (!dump_in) {
            dump_in = true;
            dump_input_tensors(idx);
        }

        try {
            LOG_DEBUG("Trying to run subrequest[" << idx << "]...");
            LOG_BLOCK();
            unsafe_run_this_prep_next(idx, next_prepared);
            job_done = true;
            LOG_DEBUG("Done: " << idx << "(exec subrequest)");
        } catch (const std::exception& ex) {
            LOG_ERROR("Subgraph [" << idx << "] - FAILED to run infer request:" << std::endl << ex.what());
            should_recreate = true;
        } catch (...) {
            LOG_ERROR("Subgraph [" << idx << "] - FAILED to run infer request: REASON UNKNOWN");
            should_recreate = true;
        }
        if (should_recreate) {
            failover = true;
            LOG_INFO("- Trying next device...");

            // Altering iterators here!! Contracts should be changed!
            comp_model_desc.device_it++;
            if (!m_npuw_model->compile_for_success(real_idx)) {
                OPENVINO_THROW("Failed to compile. No more devices are left!");
            }
            recreate_subrequests(idx);
        }
    }  // while(job_done)

    if (job_done) {
        dump_output_tensors(idx);  // FIXME: Called here unconditionally, need to refactor
        if (m_use_function_pipelining && m_funcall_pipeline[idx].next) {
            // Swap the next (pipelined, semi-prepared) infer request in the chain
            // with the default (to be accessed next) one.
            std::swap(m_subrequests[real_idx], m_funcall_pipeline[real_idx].subrequest);
        }
    }
}

void ov::npuw::JustInferRequest::unsafe_during(std::size_t real_idx, const std::function<void()>& f) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    if (!comp_model_desc.spatial) {
        // Non-spatial execution: trigger request asynchronously, run `f` in this context
        auto& r = m_subrequests[real_idx];
        r->start_async();
        f();  // expect noexcept
        r->wait();
    } else {
        // Spatial execution... Do the opposite - run f asynchronously, and meanwhile run the
        // spatial inference
        auto future = std::async(std::launch::async, f);
        unsafe_infer(real_idx);
        future.wait();
    }
}

void ov::npuw::JustInferRequest::unsafe_infer(std::size_t real_idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    auto& r = m_subrequests[real_idx];
    if (!comp_model_desc.spatial) {
        // Run normally
        r->infer();
    } else {
        // Run over the specified range... Note: the full inputs/outputs
        // must be prepared in the m_spatial_io at this point
        const auto& spatial = comp_model_desc.spatial.value();
        const auto num_outputs = comp_model_desc.compiled_model->outputs().size();

        // Create a sparse vector with full input sizes.
        // For the access simplicity, its size is aligned with function's
        // number of input parameters (activations) so some slots may be
        // not used here.
        // FIXME: All these preparations could be done statically (just once)
        std::vector<ov::Shape> full_in_shapes(comp_model_desc.param_base);
        for (auto&& param : spatial.params) {
            full_in_shapes[param.idx] = m_spatial_io[real_idx].inputs.at(param.idx)->get_shape();
        }

        // Now handle the range, even if it is not a multiply of nway (slice):
        //
        // |<- - - - full range  - - - ->|
        // +------+------+------+------+-+
        // | nway | nway | nway | nway | |
        // +------+------+------+------+-+
        //                              ^tail
        // The block is always compiled to produce nway. If we need a smaller tensor
        // on the last iteration, the sub-nway will be copied from the input range to
        // a temporary tensor, and then the sub-nwway range will be copied from the
        // request's output range.

        std::size_t offset = 0u;
        for (std::size_t i = 0u; i < spatial.nway_iters; i++, offset += spatial.nway) {
            // Collect spatial inputs for this offset
            for (auto&& param : spatial.params) {
                const auto& iport = comp_model_desc.compiled_model->inputs()[param.idx];
                r->set_tensor(
                    iport,
                    ov::npuw::util::view(m_spatial_io[real_idx].inputs.at(param.idx), param.dim, offset, spatial.nway));
            }  // for(params)

            // Now set the spatial outputs
            for (std::size_t out_idx = 0u; out_idx < num_outputs; out_idx++) {
                const auto& oport = comp_model_desc.compiled_model->outputs()[out_idx];
                r->set_tensor(oport,
                              ov::npuw::util::view(m_spatial_io[real_idx].outputs.at(out_idx),
                                                   spatial.out_dim,
                                                   offset,
                                                   spatial.nway));
            }  // for(outputs)

            // Now run the part
            r->infer();
        }  // for(full_nway_times)

        // Now process the tail, if required
        if (spatial.tail_size) {
            // Copy the sub-ranges to spatial inputs
            // NOTE: tails buffers are read from/written to at 0th offset!
            for (auto&& param : spatial.params) {
                auto in_view = ov::npuw::util::view(m_spatial_io[real_idx].inputs.at(param.idx),
                                                    param.dim,
                                                    offset,
                                                    spatial.tail_size);

                const auto& iport = comp_model_desc.compiled_model->inputs()[param.idx];
                auto out_view = ov::npuw::util::view(m_spatial_io[real_idx].input_tails.at(param.idx),
                                                     param.dim,
                                                     0,
                                                     spatial.tail_size);

                in_view->copy_to(out_view._ptr);
                r->set_tensor(iport, m_spatial_io[real_idx].input_tails.at(param.idx));
            }  // for(params)

            // Now set the tail tensors
            for (std::size_t out_idx = 0u; out_idx < num_outputs; out_idx++) {
                const auto& oport = comp_model_desc.compiled_model->outputs()[out_idx];
                r->set_tensor(oport, m_spatial_io[real_idx].output_tails.at(out_idx));
            }  // for(outputs)

            // Now run the tail infer
            r->infer();

            // Now copy the views from the output full-nway tensor to the output tensors
            for (std::size_t out_idx = 0u; out_idx < num_outputs; out_idx++) {
                const auto& oport = comp_model_desc.compiled_model->outputs()[out_idx];
                auto spatial_tensor_shape = oport.get_shape();

                auto in_view = ov::npuw::util::view(m_spatial_io[real_idx].output_tails.at(out_idx),
                                                    spatial.out_dim,
                                                    0,
                                                    spatial.tail_size);

                auto out_view = ov::npuw::util::view(m_spatial_io[real_idx].outputs.at(out_idx),
                                                     spatial.out_dim,
                                                     offset,
                                                     spatial.tail_size);
                in_view->copy_to(out_view._ptr);
            }  // for(outputs)
        }
    }
}

void ov::npuw::JustInferRequest::unsafe_run_this_prep_next(std::size_t idx, bool& next_prepared) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    const std::size_t next_idx = next(idx + 1);

    if (comp_model_desc.replaced_by) {
        // This is a function call!
        if (real_idx == real(next_idx)) {
            // The next subgraph is a call to the same function...
            // At this point, THIS infer request is already prepared.
            // Run it, then prepare it again for the next entrace
            if (m_use_function_pipelining) {
                // function pipelining is here! and the next rq is ours.
                NPUW_ASSERT(m_funcall_pipeline[idx].next.value() == next_idx);
                unsafe_during(real_idx, [&]() {
                    LOG_DEBUG("Unpacking closures for the NEXT subrequest[" << next_idx << "]...");
                    LOG_BLOCK();
                    // Note: do it here unconditionally - if this request fails,
                    // have to resubmit all the data to the recompiled pair anyway
                    bind_global_parameters(next_idx);
                    unpack_closure(next_idx, m_funcall_pipeline[real_idx].subrequest);
                });
            } else {
                // Function pipelining is not used. THIS infer request
                // is also the NEXT one. Nothing much to do here
                unsafe_infer(real_idx);
                bind_global_parameters(next_idx);
            }
        } else {
            // The next subgraph is NOT a call to the same function!
            // Trigger execution of the current one
            // FIXME: pipelining?
            if (next_idx == 0) {
                // Note: even if m_function_pipelining is ON,
                // SWAP won't happen here - see the below check for .next
                unsafe_infer(real_idx);
            } else {
                unsafe_during(real_idx, [&]() {
                    if (!next_prepared) {
                        bind_global_parameters(next_idx);
                        next_prepared = true;
                    }
                    if (m_use_function_pipelining && m_funcall_pipeline[idx].next) {
                        const auto my_next_idx = m_funcall_pipeline[idx].next.value();
                        LOG_DEBUG("Unpacking closures for the NEXT subrequest[" << my_next_idx << "]...");
                        LOG_BLOCK();
                        unpack_closure(my_next_idx, m_funcall_pipeline[real_idx].subrequest);
                    }
                });
            }
        }
    } else {
        // This is a regular subgraph. Start it async to prepare the next
        // parameters
        if (next_idx == 0) {
            unsafe_infer(real_idx);
        } else {
            unsafe_during(real_idx, [&]() {
                if (!next_prepared) {
                    bind_global_parameters(next_idx);
                    next_prepared = true;
                }
            });
        }
    }  // if (replaced_by)
}

void ov::npuw::JustInferRequest::subscribe_subrequest(std::size_t idx, Completed cb) {
    get_real_subrequest(idx)->set_callback(std::move(cb));
}

void ov::npuw::JustInferRequest::complete_subrequest(std::size_t idx) {
    // do nothing here
}

void ov::npuw::JustInferRequest::cancel_subrequest(std::size_t idx) {
    m_subrequests[idx]->cancel();
}

std::size_t ov::npuw::JustInferRequest::total_subrequests() const {
    return m_subrequests.size();
}

bool ov::npuw::JustInferRequest::supports_async_pipeline() const {
    return false;
}

void ov::npuw::JustInferRequest::update_subrequest_links(std::size_t) {
    connect_subrequests();
}
