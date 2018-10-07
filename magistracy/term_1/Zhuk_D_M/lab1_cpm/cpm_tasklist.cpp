#include "cpm.h"

#include <iostream>
#include <queue>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

void TaskList::Init() {
    initialTask = std::make_shared<Task>(STR_INIT_TASK);
    finalTask   = std::make_shared<Task>(STR_FIN_TASK);
}

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

        if (!taskInfo.second && task.IsParsed()) {
            throw std::runtime_error("A duplicate for task " + tid
                + " has been found in " + filename);
        }

        task.duration  = taskNode.second.get<int>("duration");
        task.isInitial = taskNode.second.get("isInitial", false);
        task.SetParsed();

        // there is no need to add initialTask as parent for current task
        if (task.isInitial)
            initialTask->children.emplace_back(taskInfo.first->second);
        
        BOOST_FOREACH (const ptree::value_type &childNode,
            taskNode.second.get_child("children")) {
            const std::string &cid = childNode.second.data();

            if (tid == cid)
                throw std::runtime_error("Recursive child in " + tid + " task");

            auto childInfo = tasks.emplace(cid, std::make_shared<Task>(cid));
            Task &child = *childInfo.first->second;
            task.children.emplace_back(childInfo.first->second);
            child.parents.emplace_back(taskInfo.first->second);
        }

        if (task.children.empty()) {
            finalTask->parents.emplace_back(taskInfo.first->second);
            task.children.emplace_back(finalTask);
        }
    }
}

bool TaskList::IsValid()  const {
    bool hasInitials = false;

    BOOST_FOREACH (const auto &taskNode, tasks) {
        const std::string &tid = taskNode.first;

        if (taskNode.second->duration <= 0)
            return false;
 
        if (!taskNode.second->isInitial && taskNode.second->parents.empty())
            return false;

        hasInitials |= taskNode.second->isInitial;

        BOOST_FOREACH (const std::shared_ptr<Task> &child, taskNode.second->children) {
            if (child->taskId == STR_FIN_TASK)
                continue;

            if (tasks.find(child->taskId) == tasks.end())
                return false;
        }
    }

    return hasInitials;
}

void TaskList::ForwardTaskLookup() {
    std::queue<std::shared_ptr<Task>> q;

    BOOST_FOREACH(const std::shared_ptr<Task> &task, initialTask->children)
        q.push(task);

    while(!q.empty()) {
        std::shared_ptr<Task> &curr = q.front();
        int timestamp = curr->earliestStart + curr->duration;

        BOOST_FOREACH(std::shared_ptr<Task> &child, curr->children) {
            child->earliestStart = std::max(child->earliestStart, timestamp);
            q.push(child);
        }

        q.pop();
    }

    finalTask->latestFinish = finalTask->earliestStart;
}

void TaskList::BackwardTaskLookup() {
    std::queue<std::shared_ptr<Task>> q;

    BOOST_FOREACH(const std::shared_ptr<Task> &task, finalTask->parents) {
        task->latestFinish = finalTask->latestFinish;
        q.push(task);
    }

    while(!q.empty()) {
        std::shared_ptr<Task> &curr = q.front();
        int timestamp = curr->latestFinish - curr->duration;
        curr->reserve = timestamp - curr->earliestStart;

        if (curr->reserve == 0) {
            criticalPath.emplace_front(curr);
            criticalPathDuration += curr->duration;
        }

        BOOST_FOREACH(std::shared_ptr<Task> &parent, curr->parents) {
            if (parent->latestFinish > 0)
                parent->latestFinish = std::min(parent->latestFinish, timestamp);
            else
                parent->latestFinish = timestamp;

            q.push(parent);
        }

        q.pop();
    }
}

void TaskList::FindCriticalPath() {
    ForwardTaskLookup();
    BackwardTaskLookup();
}

void TaskList::Dump(const std::string &suffix) const {
    std::cout << "--------------- TaskList: " << suffix
        << " ---------------" << std::endl;

    BOOST_FOREACH (const auto &taskNode, tasks) {
        const std::vector<std::shared_ptr<Task>> &children = taskNode.second->children;
        const std::vector<std::shared_ptr<Task>> &parents  = taskNode.second->parents;

        std::cout << "id = " << taskNode.first
            << ":\n\tisInitial = " << taskNode.second->isInitial
            << ";\n\tduration = " << taskNode.second->duration
            << ";\n\tearliestStart = " << taskNode.second->earliestStart
            << ";\n\tlatestFinish = " << taskNode.second->latestFinish
            << ";\n\treserve = " << taskNode.second->reserve
            << ";\n\tchildren = [";

        for (size_t i = 0; i < children.size(); ++i) {
            if (children[i]->taskId == STR_FIN_TASK)
                continue;

            std::cout << children[i]->taskId
                << ((i < children.size() - 1) ? "; " : "");
        }

        std::cout << "];\n\tparents = [";

        for (size_t i = 0; i < parents.size(); ++i) {
            std::cout << parents[i]->taskId
                << ((i < parents.size() - 1) ? "; " : "");
        }

        std::cout << "];" << std::endl;
    }
}

void TaskList::DumpCriticalPath() {
    std::cout << "----------- Critical Path -----------" << std::endl;

    for (size_t i = 0; i < criticalPath.size(); ++i) {
        std::cout << criticalPath[i]->taskId
            << ((i < criticalPath.size() - 1) ? " --> " : "");
    }

    std::cout << ";\tduration = " << criticalPathDuration << std::endl;
}
