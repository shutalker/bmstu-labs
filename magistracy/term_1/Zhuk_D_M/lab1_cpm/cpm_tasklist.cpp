#include "cpm.h"

#include <iostream>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

void TaskList::LoadFromJsonFile(const std::string &filename) {
    namespace fs = boost::filesystem;
    using boost::property_tree::ptree;

    if (!fs::exists(filename) || !fs::is_regular_file(filename))
        throw std::runtime_error("No such file: " + filename);

    ptree tree;
    read_json(filename, tree);

    BOOST_FOREACH (const ptree::value_type &taskNode, tree) {
        const std::string &tid = taskNode.second.get<std::string>("taskId");
        auto taskInfo = tasks.emplace(tid, std::make_shared<Task>(tid));
        Task &task = *taskInfo.first->second;

        if (!taskInfo.second && task.IsInitialized()) {
            throw std::runtime_error("A duplicate for task " + tid
                + " has been found in " + filename);
        }

        task.duration  = taskNode.second.get<int>("duration");
        task.isInitial = taskNode.second.get("isInitial", false);
        task.SetInitialized();

        if (task.isInitial)
            initialTasks.emplace_back(taskInfo.first->second);
        
        BOOST_FOREACH (const ptree::value_type &childNode,
            taskNode.second.get_child("children")) {
            const std::string &cid = childNode.second.data();

            if (tid == cid)
                throw std::runtime_error("Recursive child in " + tid + " task");

            auto childInfo = tasks.emplace(cid, std::make_shared<Task>(cid));
            task.children.emplace_back(childInfo.first->second);
            reverseTaskLinks[cid].emplace_back(taskInfo.first->second);
        }
    }
}

bool TaskList::IsValid()  const {
    bool hasInitials = false;

    BOOST_FOREACH (const auto &taskNode, tasks) {
        const std::string &tid = taskNode.first;

        if (taskNode.second->duration <= 0)
            return false;
 
        if (!taskNode.second->isInitial
            && reverseTaskLinks.find(tid) == reverseTaskLinks.end())
            return false;

        hasInitials |= taskNode.second->isInitial;

        BOOST_FOREACH (const std::shared_ptr<Task> &child, taskNode.second->children) {
            if (tasks.find(child->taskId) == tasks.end())
                return false;
        }
    }

    return hasInitials;
}

const std::vector<std::shared_ptr<Task>> & TaskList::GetInitialTasks() const {
    return initialTasks;
}

void TaskList::Dump() const {
    std::cout << "--------------- TaskList ---------------" << std::endl;

    BOOST_FOREACH (const auto &taskNode, tasks) {
        const std::vector<std::shared_ptr<Task>> &children = taskNode.second->children;

        std::cout << "id = " << taskNode.first
            << ";\tisInitial = " << taskNode.second->isInitial
            << ";\tduration = " << taskNode.second->duration
            << ";\treserve = " << taskNode.second->reserve
            << ";\tchildren = [";

        for (size_t i = 0; i < children.size(); ++i) {
            std::cout << children[i]->taskId
                << ((i < children.size() - 1) ? "; " : "");
        }

        std::cout << "]" << std::endl;
    }
}

void TaskList::DumpReverseTaskLinks() const {
    std::cout << "----------- ReverseTaskLinks -----------" << std::endl;

    BOOST_FOREACH (const auto &node, reverseTaskLinks) {
        const std::vector<std::shared_ptr<Task>> &parents = node.second;
        std::cout << "child = " << node.first << ";\tparents: [";

        for (size_t i = 0; i < parents.size(); ++i) {
            std::cout << parents[i]->taskId
                << ((i < parents.size() - 1) ? "; " : "");
        }

        std::cout << "]" << std::endl;
    }
}
