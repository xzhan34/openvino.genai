#include <queue>

#include "module.hpp"

namespace ov {
namespace genai {
namespace module {
PipelineModuleInstance sort_pipeline(PipelineModuleInstance& pipeline_instrance) {
    std::unordered_map<std::string, IBaseModule::PTR> module_map;
    // 1. Init graph and in-degrees (preparation for Kahn's algorithm)
    for (const auto& module_ptr : pipeline_instrance) {
        module_map[module_ptr->get_module_name()] = module_ptr;
    }

    // Current module name -> List of subsequent modules
    std::unordered_map<std::string, std::vector<std::string>> adjacency_list;
    // Module name -> input module number
    std::unordered_map<std::string, int> in_degree;
    for (const auto& pair : module_map) {
        in_degree[pair.first] = pair.second->inputs.size();
        adjacency_list[pair.first] = {};
        for (auto& output : pair.second->outputs) {
            adjacency_list[pair.first].push_back(output.module_ptr->get_module_name());
        }
    }

    // 2. Init ready queue (Modules with in_degree are 0)
    std::queue<std::string> ready_queue;
    for (const auto& pair : in_degree) {
        if (pair.second == 0) {
            ready_queue.push(pair.first);
        }
    }

    // 3. Topological sorting
    PipelineModuleInstance sorted_pipeline;

    while (!ready_queue.empty()) {
        std::string current_name = ready_queue.front();
        ready_queue.pop();

        sorted_pipeline.push_back(module_map.at(current_name));

        // Iterate through all subsequent dependencies (neighbors) of the current module.
        for (const std::string& neighbor_name : adjacency_list[current_name]) {
            // 1. Reduce in_degree
            in_degree[neighbor_name]--;

            // 2. Check the ready state: If the in-degree becomes 0, add it to the ready queue.
            if (in_degree[neighbor_name] == 0) {
                ready_queue.push(neighbor_name);
            }
        }
    }

    // 5. Check for circular dependencies.
    OPENVINO_ASSERT(
        sorted_pipeline.size() == pipeline_instrance.size(),
        "The pipeline contains a circular dependency (Cycle Detected). Please check input pipeline config.");

    return sorted_pipeline;
}
}  // namespace module
}  // namespace genai
}  // namespace ov